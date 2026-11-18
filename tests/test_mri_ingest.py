"""Tests for MRI NIfTI ingestion (scripts/preprocessing/preprocess_mri.py)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "preprocessing")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from PIL import Image

from mvp_combine_indices import main as combine_main
from mvp_prepare_data import index_csv_to_manifest
from preprocess_mri import (
    detect_sequence,
    derive_patient_id,
    load_volume,
    main as mri_main,
    process_volume,
)
from zoo.encoding import u16_to_zscore, zscore_encoding_token
from zoo.registry import DatasetRegistry

SPACING = (0.8, 0.9, 2.0)
SHAPE_RAS = (12, 14, 6)
BRIGHT_S_MIN = 9.0
MARKER_S_MIN = 9.0


def _world_value(world_x: float, world_y: float, world_z: float) -> float:
    if world_z >= BRIGHT_S_MIN and world_x <= 0.5 and world_y <= 0.5:
        return 1000.0
    if world_z >= BRIGHT_S_MIN:
        return 100.0
    return 1.0


def _write_volume(path: Path, voxel_axes: tuple[int, ...], world_dirs: tuple[tuple[int, int], ...]) -> None:
    mat = np.zeros((3, 3))
    for v_ax, (w_ax, sign) in enumerate(world_dirs):
        mat[w_ax, v_ax] = sign * SPACING[w_ax]
    offsets = np.zeros(3)
    for v_ax, (w_ax, sign) in enumerate(world_dirs):
        if sign < 0:
            offsets[w_ax] = (voxel_axes[v_ax] - 1) * SPACING[w_ax]
    affine = np.eye(4)
    affine[:3, :3] = mat
    affine[:3, 3] = offsets
    data = np.empty(voxel_axes, dtype=np.float32)
    it = np.nditer(np.empty(voxel_axes, dtype=np.int8), flags=["multi_index"])
    while not it.finished:
        i, j, k = it.multi_index
        world = affine @ np.array([i, j, k, 1.0])
        data[i, j, k] = _world_value(world[0], world[1], world[2])
        it.iternext()
    nib.Nifti1Image(data, affine).to_filename(str(path))


def _ras_volume(path: Path) -> None:
    _write_volume(path, SHAPE_RAS, ((0, 1), (1, 1), (2, 1)))


def _lps_volume(path: Path) -> None:
    _write_volume(path, SHAPE_RAS, ((0, -1), (1, -1), (2, 1)))


def _permuted_volume(path: Path) -> None:
    _write_volume(path, (SHAPE_RAS[2], SHAPE_RAS[0], SHAPE_RAS[1]), ((2, 1), (0, 1), (1, 1)))


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _decoded_slice(png_path: Path) -> np.ndarray:
    arr = np.array(Image.open(png_path))
    return u16_to_zscore(arr)


def test_detect_sequence_brats():
    assert detect_sequence("BraTS_001_1_0000") == ("BraTS_001_1", "t1")
    assert detect_sequence("BraTS_001_1_0001") == ("BraTS_001_1", "t1ce")
    assert detect_sequence("BraTS_001_1_0002") == ("BraTS_001_1", "t2")
    assert detect_sequence("BraTS_001_1_0003") == ("BraTS_001_1", "t2flair")


def test_detect_sequence_named_tokens():
    assert detect_sequence("amos_0001_t2") == ("amos_0001", "t2")
    assert detect_sequence("knee_scan_9_pdfs") == ("knee_scan_9", "pdfs")
    assert detect_sequence("brain_12_T2w") == ("brain_12", "t2w")


def test_detect_sequence_none_when_unrecognized():
    assert detect_sequence("BraTS_001_1") == ("BraTS_001_1", None)


def test_derive_patient_id():
    assert derive_patient_id("BraTS_001_1") == "BraTS_001"
    assert derive_patient_id("BraTS_001_2") == "BraTS_001"
    assert derive_patient_id("BraTS_002_1") == "BraTS_002"
    assert derive_patient_id("amos_0001") == "amos_0001"
    assert derive_patient_id("amos_0001") != derive_patient_id("amos_0002")
    assert derive_patient_id("knee_scan_9") == "knee_scan_9"


def test_load_volume_reorients_to_ras_axial():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "v.nii.gz"
        _permuted_volume(p)
        volume, spacing = load_volume(p)
        assert volume.shape == (SHAPE_RAS[2], SHAPE_RAS[1], SHAPE_RAS[0])
        assert spacing == pytest.approx(SPACING)
        assert volume[-1].max() > volume[0].max()


def test_per_volume_zscore_normalization(tmp_path):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    _ras_volume(vol_dir / "amos_0001_t1.nii.gz")
    out_root = tmp_path / "processed"

    rc = mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "amos-mri",
    ])
    assert rc == 0

    rows = _read_index(out_root / "amos-mri" / "index.csv")
    assert len(rows) == SHAPE_RAS[2]
    decoded = [_decoded_slice(Path(r["png_path"])) for r in rows]
    stacked = np.stack(decoded)
    assert abs(float(stacked.mean())) < 1e-2
    assert abs(float(stacked.std()) - 1.0) < 1e-2
    assert np.argmax(decoded[-1]) == 0


def test_index_row_fields(tmp_path):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    _ras_volume(vol_dir / "amos_0001_t1.nii.gz")
    out_root = tmp_path / "processed"
    assert mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "amos-mri",
    ]) == 0

    rows = _read_index(out_root / "amos-mri" / "index.csv")
    row = rows[0]
    assert row["encoding"] == zscore_encoding_token()
    assert row["sequence"] == "t1"
    assert row["patient_id"] == "amos_0001"
    assert row["series_dir"].endswith("amos_0001_t1")
    assert float(row["spacing_x"]) == pytest.approx(SPACING[0])
    assert float(row["spacing_y"]) == pytest.approx(SPACING[1])
    assert float(row["spacing_z"]) == pytest.approx(SPACING[2])


def test_orientation_invariance(tmp_path):
    out_root = tmp_path / "processed"
    for builder in (_ras_volume, _lps_volume, _permuted_volume):
        vol_dir = tmp_path / "raw" / builder.__name__
        vol_dir.mkdir(parents=True)
        builder(vol_dir / "amos_0001_t1.nii.gz")
        rc = mri_main([
            "--nifti-dir", str(vol_dir),
            "--out-root", str(out_root),
            "--dataset-name", f"amos-{builder.__name__}",
        ])
        assert rc == 0
        rows = _read_index(out_root / f"amos-{builder.__name__}" / "index.csv")
        decoded = [_decoded_slice(Path(r["png_path"])) for r in rows]
        assert len(decoded) == SHAPE_RAS[2]
        assert decoded[-1][0, 0] >= np.max(decoded[-1]) - 1e-6
        for d in decoded[:-1]:
            assert decoded[-1][0, 0] > d[0, 0]


def test_brats_multi_sequence_and_patient_grouping(tmp_path):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    for suffix in ("0000", "0001", "0002", "0003"):
        _ras_volume(vol_dir / f"BraTS_001_1_{suffix}.nii.gz")
    _ras_volume(vol_dir / "BraTS_001_2_0000.nii.gz")
    _ras_volume(vol_dir / "BraTS_002_1_0001.nii.gz")
    _ras_volume(vol_dir / "BraTS_001_1.nii.gz")
    out_root = tmp_path / "processed"

    rc = mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "brats-2021",
    ])
    assert rc == 0
    rows = _read_index(out_root / "brats-2021" / "index.csv")

    series = sorted({r["series_dir"] for r in rows})
    assert len(series) == 6
    patients = {r["series_dir"]: r["patient_id"] for r in rows}
    assert set(patients.values()) == {"BraTS_001", "BraTS_002"}
    seqs = {r["series_dir"]: r["sequence"] for r in rows}
    assert set(seqs.values()) == {"t1", "t1ce", "t2", "t2flair"}
    assert not any("BraTS_001_1.nii" in s or s.endswith("BraTS_001_1") for s in series)


def test_unrecognized_sequence_skipped_and_counted(tmp_path, capsys):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    _ras_volume(vol_dir / "BraTS_001_1.nii.gz")
    out_root = tmp_path / "processed"

    rc = mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "brats-2021",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "unrecognized_sequence" in out


def test_single_sequence_mode(tmp_path):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    _lps_volume(vol_dir / "amos_0001.nii.gz")
    out_root = tmp_path / "processed"

    rc = mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "amos-mri",
        "--single-sequence", "mri",
    ])
    assert rc == 0
    rows = _read_index(out_root / "amos-mri" / "index.csv")
    assert rows[0]["sequence"] == "mri"
    assert rows[0]["patient_id"] == "amos_0001"


def test_rerun_reindexes_with_real_spacing(tmp_path):
    vol_dir = tmp_path / "raw"
    vol_dir.mkdir()
    _ras_volume(vol_dir / "amos_0001_t1.nii.gz")
    out_root = tmp_path / "processed"
    assert mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "amos-mri",
    ]) == 0
    first = _read_index(out_root / "amos-mri" / "index.csv")

    assert mri_main([
        "--nifti-dir", str(vol_dir),
        "--out-root", str(out_root),
        "--dataset-name", "amos-mri",
    ]) == 0
    second = _read_index(out_root / "amos-mri" / "index.csv")
    assert len(second) == len(first)
    assert float(second[0]["spacing_z"]) == pytest.approx(SPACING[2])


def test_missing_dir_exits_2(tmp_path, capsys):
    rc = mri_main([
        "--nifti-dir", str(tmp_path / "nope"),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "amos-mri",
    ])
    assert rc == 2


def test_empty_dir_exits_2(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = mri_main([
        "--nifti-dir", str(empty),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "amos-mri",
    ])
    assert rc == 2


def _write_mri_index(path: Path, dataset: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "png_path", "series_dir", "slice_index", "encoding",
            "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
        ])
        w.writerow(["x.png", f"{dataset}_v1_t1", 0, zscore_encoding_token(),
                    "0.8", "0.8", "2.0", "v1", "t1"])


def _write_ct_index(path: Path, dataset: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z"])
        w.writerow(["y.png", f"{dataset}_s1", 0,
                    "hu16_i16_offset32768_clip-1000_4000", "0.7", "0.7", "1.5"])


def test_combine_passes_mri_columns(tmp_path, monkeypatch):
    a = tmp_path / "a_index.csv"
    b = tmp_path / "b_index.csv"
    _write_mri_index(a, "amos-mri")
    _write_mri_index(b, "brats-2021")
    out = tmp_path / "combined" / "index.csv"
    monkeypatch.setattr(sys, "argv", [
        "combine_indices.py",
        "--inputs", f"amos-mri:{a}",
        "--inputs", f"brats-2021:{b}",
        "--out", str(out),
    ])
    assert combine_main() == 0
    rows = _read_index(out)
    assert {r["patient_id"] for r in rows} == {"v1"}
    assert {r["sequence"] for r in rows} == {"t1"}


def test_combine_refuses_mixed_encoding_families(tmp_path, monkeypatch, capsys):
    mri_csv = tmp_path / "mri_index.csv"
    ct_csv = tmp_path / "ct_index.csv"
    _write_mri_index(mri_csv, "amos-mri")
    _write_ct_index(ct_csv, "lidc-idri")
    out = tmp_path / "combined" / "index.csv"
    monkeypatch.setattr(sys, "argv", [
        "combine_indices.py",
        "--inputs", f"amos-mri:{mri_csv}",
        "--inputs", f"lidc-idri:{ct_csv}",
        "--out", str(out),
    ])
    assert combine_main() != 0
    err = capsys.readouterr().err
    assert "encoding" in err
    assert not out.exists()


def test_index_csv_to_manifest_keeps_patient_id(tmp_path):
    idx = tmp_path / "index.csv"
    _write_mri_index(idx, "amos-mri")
    manifest = index_csv_to_manifest(idx, "amos-mri", ["abdomen"])
    record = manifest.records[0]
    assert record.patient_id == "v1"
    assert record.dataset == "amos-mri"


def test_real_catalog_has_mri_entries():
    registry = DatasetRegistry(_REPO_ROOT / "zoo" / "datasets")
    mri = registry.query(modality="mri")
    names = {e.name for e in mri}
    assert {"amos-mri", "brats-2021", "fastmri"} <= names
    for entry in mri:
        assert entry.license
        assert entry.citation
        assert entry.preprocessing.encoding == "zscore16"

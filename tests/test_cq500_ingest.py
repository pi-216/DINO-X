"""Tests for CQ500 ingestion (scripts/preprocessing/preprocess_cq500.py)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "preprocessing")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

pyd = pytest.importorskip("pydicom")

from mvp_combine_indices import main as combine_main
from preprocess_cq500 import main as cq500_main
from zoo.encoding import hu_encoding_token, u16_to_hu
from zoo.manifest import DataManifest
from zoo.registry import DatasetRegistry

HU_TOL = 0.05


def _write_ct(
    path: Path,
    stored: np.ndarray,
    study: str = "1.2.276.0.7230010.3.1.2.296485376.1.1",
    series: str = "1.2.276.0.7230010.3.1.3.296485376.1.1",
    sop: str | None = None,
    patient: str | None = "CQ500-CT-0",
    ipp_z: float | None = 0.0,
    spacing: tuple[str, str] = ("0.451172", "0.451172"),
    slice_thickness: str = "5.0",
    spacing_between: str | None = "10.125",
    slope: float = 1.0,
    intercept: float = -1024.0,
    modality: str = "CT",
    instance: int = 1,
) -> None:
    ds = pyd.Dataset()
    ds.SOPClassUID = pyd.uid.CTImageStorage
    ds.SOPInstanceUID = sop or pyd.uid.generate_uid()
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.Modality = modality
    if patient is not None:
        ds.PatientID = patient
    ds.InstanceNumber = str(instance)
    if ipp_z is not None:
        ds.ImagePositionPatient = [0.0, 0.0, ipp_z]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = list(spacing)
    ds.SliceThickness = slice_thickness
    if spacing_between is not None:
        ds.SpacingBetweenSlices = spacing_between
    ds.RescaleSlope = str(slope)
    ds.RescaleIntercept = str(intercept)
    ds.Rows = stored.shape[0]
    ds.Columns = stored.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = stored.astype("<u2").tobytes()
    meta = pyd.FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = pyd.uid.ExplicitVRLittleEndian
    ds.file_meta = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def _const(value: int, size: int = 8) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint16)


def _series_args(*, study: str, series: str, patient: str) -> dict:
    return {"study": study, "series": series, "patient": patient}


def _read_index(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_png_u16(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path))


def _read_png_hu(path: Path) -> float:
    return float(u16_to_hu(_read_png_u16(path))[0][0])


@pytest.fixture()
def cq500_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    study = "1.2.276.0.7230010.3.1.2.296485376.1.1"
    thick = "1.2.276.0.7230010.3.1.3.296485376.1.101"
    thin = "1.2.276.0.7230010.3.1.3.296485376.1.102"
    base = raw / "qct01" / "CQ500CT0 CQ500CT0" / "Unknown Study"
    thick_files = {0: (10.125, 25, 18), 1: (0.0, 24, 17), 2: (20.25, 26, 19)}
    for i, (z, stored, inst) in thick_files.items():
        _write_ct(
            base / "CT Plain" / f"CT00000{i}.dcm", _const(stored),
            ipp_z=z, instance=inst, spacing_between="9.9",
            **_series_args(study=study, series=thick, patient="CQ500-CT-0"),
        )
    for i in range(3):
        _write_ct(
            base / "CT Thin Plain" / f"CT00010{i}.dcm", _const(2024),
            ipp_z=2.0 * i, instance=i + 1, slice_thickness="1.0",
            spacing_between="1.9",
            **_series_args(study=study, series=thin, patient="CQ500-CT-0"),
        )
    other = raw / "qct01" / "CQ500CT1 CQ500CT1" / "Unknown Study" / "CT Plain"
    for i in range(3):
        _write_ct(
            other / f"CT00020{i}.dcm", _const(700), ipp_z=5.0 * i,
            instance=i + 1,
            study="1.2.276.0.7230010.3.1.2.296485376.1.2",
            series="1.2.276.0.7230010.3.1.3.296485376.1.201",
            patient="CQ500-CT-1",
        )
    (raw / "labels.csv").write_text("Patient,Any\n1,0\n")
    return raw


def test_nested_cq500_layout_groups_by_header_uids(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert len(rows) == 9
    series_dirs = {r["series_dir"] for r in rows}
    assert len(series_dirs) == 3
    counts = sorted(sum(1 for r in rows if r["series_dir"] == s) for s in series_dirs)
    assert counts == [3, 3, 3]


def test_orders_slices_by_imagepositionpatient(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    thick_rows = [
        r for r in rows if r["series_dir"].endswith(
            "296485376.1.1_1.2.276.0.7230010.3.1.3.296485376.1.101"
        )
    ]
    assert [int(r["slice_index"]) for r in thick_rows] == [0, 1, 2]
    assert [_read_png_hu(Path(r["png_path"])) for r in thick_rows] == pytest.approx(
        [-1000.0, -999.0, -998.0], abs=HU_TOL
    )


def test_slice_bytes_use_canonical_hu16_scale(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    air_row = next(
        r for r in rows if r["series_dir"].endswith(".1.1_1.2.276.0.7230010.3.1.3.296485376.1.101")
    )
    bone_row = next(
        r for r in rows if r["series_dir"].endswith(".1.1_1.2.276.0.7230010.3.1.3.296485376.1.102")
    )
    air_px = _read_png_u16(Path(air_row["png_path"]))[0][0]
    bone_px = _read_png_u16(Path(bone_row["png_path"]))[0][0]
    assert int(air_px) == 22768
    assert int(bone_px) == 42768


def test_applies_rescale_slope_intercept(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(
            raw / f"s{i}.dcm", _const(1000), ipp_z=1.0 * i,
            slope=0.5, intercept=-1024.0,
            study="1.2.3.1", series="1.2.3.2", patient="CQ500-CT-9",
        )
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert all(abs(_read_png_hu(Path(r["png_path"])) - -524.0) < HU_TOL for r in rows)


def test_z_spacing_uses_ipp_deltas_over_thickness(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    thick_rows = [
        r for r in rows if r["series_dir"].endswith(".1.1_1.2.276.0.7230010.3.1.3.296485376.1.101")
    ]
    assert all(abs(float(r["spacing_z"]) - 10.125) < 1e-6 for r in thick_rows)
    assert all(abs(float(r["spacing_x"]) - 0.451172) < 1e-6 for r in thick_rows)
    assert all(abs(float(r["spacing_y"]) - 0.451172) < 1e-6 for r in thick_rows)


def test_z_spacing_falls_back_to_spacing_between_slices(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(
            raw / f"n{i}.dcm", _const(500), ipp_z=None, instance=i + 1,
            slice_thickness="5.0", spacing_between="7.5",
            study="1.2.3.1", series="1.2.3.2", patient="CQ500-CT-8",
        )
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert all(abs(float(r["spacing_z"]) - 7.5) < 1e-6 for r in rows)


def test_patient_id_from_header_with_study_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(raw / f"a{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.3.4", series="1.2.3.5", patient="CQ500-CT-77")
    for i in range(3):
        _write_ct(raw / f"b{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.3.9", series="1.2.3.8", patient=None)
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    by_study = {r["series_dir"].split("_")[1]: r["patient_id"] for r in rows}
    assert by_study["1.2.3.4"] == "CQ500-CT-77"
    assert by_study["1.2.3.9"] == "1.2.3.9"


def test_skips_series_below_min_slices(tmp_path: Path, cq500_raw: Path) -> None:
    raw = cq500_raw / "qct01" / "CQ500CT2 CQ500CT2" / "Unknown Study" / "CT 1slice"
    for i in range(2):
        _write_ct(raw / f"t{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.276.0.7230010.3.1.2.296485376.1.3",
                  series="1.2.276.0.7230010.3.1.3.296485376.1.301",
                  patient="CQ500-CT-2")
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert all("296485376.1.301" not in r["series_dir"] for r in rows)
    assert len(rows) == 9
    out2 = tmp_path / "processed2"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out2),
                       "--min-slices", "2"]) == 0
    rows2 = _read_index(out2 / "cq500" / "index.csv")
    assert any("296485376.1.301" in r["series_dir"] for r in rows2)


def test_skips_non_ct_series(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(raw / f"ct{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.3.1", series="1.2.3.2", patient="p1")
    for i in range(3):
        _write_ct(raw / f"mr{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.3.3", series="1.2.3.4", patient="p2", modality="MR")
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert len(rows) == 3
    assert all(r["series_dir"].count("1.2.3.2") == 1 for r in rows)


def test_all_non_ct_exits_one(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(raw / f"mr{i}.dcm", _const(500), ipp_z=float(i),
                  study="1.2.3.3", series="1.2.3.4", patient="p2", modality="MR")
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 1


def test_dedupes_repeated_sop_instance_uid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "s0.dcm", _const(500), sop="1.2.3.100", ipp_z=0.0,
              study="1.2.3.1", series="1.2.3.2", patient="p1")
    _write_ct(raw / "z-s0-copy.dcm", _const(900), sop="1.2.3.100", ipp_z=0.0,
              study="1.2.3.1", series="1.2.3.2", patient="p1")
    for i in (1, 2):
        _write_ct(raw / f"s{i}.dcm", _const(600 + i), sop=f"1.2.3.10{i}", ipp_z=float(i),
                  study="1.2.3.1", series="1.2.3.2", patient="p1")
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert len(rows) == 3
    assert abs(_read_png_hu(Path(rows[0]["png_path"])) - (500 - 1024)) < HU_TOL


def test_rerun_resumes_from_meta_without_rewriting_slices(
    tmp_path: Path, cq500_raw: Path
) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    first = _read_index(out / "cq500" / "index.csv")
    png = next((out / "cq500").rglob("slice_0000.png"))
    mtime = png.stat().st_mtime_ns
    meta = json.loads((png.parent / "meta.json").read_text())
    assert meta["encoding"] == hu_encoding_token()
    assert meta["patient_id"].startswith("CQ500-CT-")
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    second = _read_index(out / "cq500" / "index.csv")
    assert len(second) == len(first)
    assert png.stat().st_mtime_ns == mtime


def test_index_columns_and_encoding_token(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "cq500" / "index.csv")
    assert list(rows[0].keys()) == [
        "png_path", "series_dir", "slice_index", "encoding",
        "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
    ]
    assert all(r["encoding"] == hu_encoding_token() for r in rows)
    assert all(r["sequence"] == "ct" for r in rows)


def test_manifest_out_writes_parquet_with_identity(tmp_path: Path, cq500_raw: Path) -> None:
    out = tmp_path / "processed"
    manifest_path = tmp_path / "manifests" / "cq500.parquet"
    rc = cq500_main([
        "--dicom-dir", str(cq500_raw), "--out-root", str(out),
        "--manifest-out", str(manifest_path),
    ])
    assert rc == 0
    manifest = DataManifest.load(manifest_path)
    assert len(manifest) == 9
    assert all(r.patient_id in {"CQ500-CT-0", "CQ500-CT-1"} for r in manifest.records)
    assert all(r.dataset == "cq500" for r in manifest.records)
    assert all("brain" in r.organs_present for r in manifest.records)
    thick = [r for r in manifest.records if r.series_id.endswith(".1.1_1.2.276.0.7230010.3.1.3.296485376.1.101")]
    assert all(abs(r.slice_thickness - 10.125) < 1e-6 for r in thick)
    assert all(abs(r.pixel_spacing_x - 0.451172) < 1e-6 for r in thick)


def test_combine_accepts_ct_and_refuses_mri(
    tmp_path: Path, cq500_raw: Path, monkeypatch
) -> None:
    out = tmp_path / "processed"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out)]) == 0
    cq_index = out / "cq500" / "index.csv"
    mri_index = tmp_path / "mri_index.csv"
    mri_index.write_text(
        "png_path,series_dir,slice_index,encoding,spacing_x,spacing_y,spacing_z,patient_id,sequence\n"
        "x/slice_0000.png,br_x,0,zscore16_u16_offset32768_scale100_clip-300_300,"
        "1.0,1.0,1.0,pat,t1\n"
    )
    ok_out = tmp_path / "combined_ok.csv"
    monkeypatch.setattr(sys, "argv", [
        "mvp_combine_indices.py",
        "--inputs", f"cq500:{cq_index}",
        "--inputs", f"lidc:{cq_index}",
        "--out", str(ok_out),
    ])
    assert combine_main() == 0
    bad_out = tmp_path / "combined_bad.csv"
    monkeypatch.setattr(sys, "argv", [
        "mvp_combine_indices.py",
        "--inputs", f"cq500:{cq_index}",
        "--inputs", f"brats:{mri_index}",
        "--out", str(bad_out),
    ])
    assert combine_main() == 1


def test_registry_entry_uses_canonical_encoding() -> None:
    registry = DatasetRegistry(_REPO_ROOT / "zoo" / "datasets")
    entry = registry.get("cq500")
    assert entry is not None
    assert entry.modality == "ct"
    assert entry.preprocessing.encoding == "hu16"
    assert entry.preprocessing.index_csv == "index.csv"
    assert entry.total_slices > 0
    assert entry.total_series > 0
    raw = (_REPO_ROOT / "zoo" / "datasets" / "ct" / "cq500.yaml").read_text()
    block = raw.split("preprocessing:", 1)[1].split("citation:", 1)[0]
    assert "encoding: hu16" in block
    assert "hu_shift" not in block
    assert "scale:" not in block


def test_workers_produce_index_identical_to_serial(tmp_path: Path, cq500_raw: Path) -> None:
    out1 = tmp_path / "p1"
    out2 = tmp_path / "p2"
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out1),
                       "--workers", "1"]) == 0
    assert cq500_main(["--dicom-dir", str(cq500_raw), "--out-root", str(out2),
                       "--workers", "4"]) == 0
    rows1 = _read_index(out1 / "cq500" / "index.csv")
    rows2 = _read_index(out2 / "cq500" / "index.csv")
    assert [(r["series_dir"], r["slice_index"], r["spacing_z"]) for r in rows1] == \
           [(r["series_dir"], r["slice_index"], r["spacing_z"]) for r in rows2]
    assert len(rows1) == len(rows2) == 9


def test_main_exit_codes_on_bad_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert cq500_main(["--dicom-dir", str(missing), "--out-root", str(tmp_path / "p")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.md").write_text("nothing here")
    assert cq500_main(["--dicom-dir", str(empty), "--out-root", str(tmp_path / "p")]) == 2

"""Tests for RSNA ICH ingestion (scripts/preprocessing/preprocess_rsna_ich.py)."""

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
from preprocess_rsna_ich import main as rsna_main
from zoo.encoding import hu_encoding_token, u16_to_hu
from zoo.manifest import DataManifest
from zoo.registry import DatasetRegistry

HU_TOL = 0.15


def _write_ct(
    path: Path,
    pixels: np.ndarray,
    study: str = "1.2.826.0.1.3680043.10.1001",
    series: str = "1.2.826.0.1.3680043.10.2001",
    sop: str | None = None,
    patient: str | None = "00020d3ce",
    ipp_z: float = 0.0,
    spacing: tuple[str, str] = ("0.6", "0.7"),
    slice_thickness: str = "1.0",
    slope: float = 0.5,
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
    ds.ImagePositionPatient = [0.0, 0.0, ipp_z]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = list(spacing)
    ds.SliceThickness = slice_thickness
    ds.RescaleSlope = str(slope)
    ds.RescaleIntercept = str(intercept)
    ds.Rows = pixels.shape[0]
    ds.Columns = pixels.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype("<u2").tobytes()
    meta = pyd.FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = pyd.uid.ExplicitVRLittleEndian
    ds.file_meta = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def _const(value: int, size: int = 8) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint16)


def _read_index(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_png_hu(path: Path) -> float:
    from PIL import Image

    arr = np.array(Image.open(path))
    return float(u16_to_hu(arr)[0][0])


@pytest.fixture()
def rsna_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    sop1 = "1.2.826.0.1.3680043.10.9001"
    sop2 = "1.2.826.0.1.3680043.10.9002"
    sop3 = "1.2.826.0.1.3680043.10.9003"
    study = "1.2.826.0.1.3680043.10.1001"
    series = "1.2.826.0.1.3680043.10.2001"
    _write_ct(
        raw / "9z10.dcm", _const(2500, 8), study=study, series=series,
        sop=sop3, ipp_z=10.0, instance=3,
    )
    _write_ct(
        raw / "5z0.dcm", _const(500, 8), study=study, series=series,
        sop=sop1, ipp_z=0.0, instance=1,
    )
    _write_ct(
        raw / "0z5.dcm", _const(1500, 8), study=study, series=series,
        sop=sop2, ipp_z=5.0, instance=2,
    )
    _write_ct(
        raw / "other9z55.dcm", _const(1000, 8), study=study,
        series="1.2.826.0.1.3680043.10.2002",
        ipp_z=55.0,
    )
    (raw / "labels.csv").write_text("Id,Label\n1_any,0\n")
    return raw


def test_groups_flat_files_by_study_and_series(tmp_path: Path, rsna_raw: Path) -> None:
    out = tmp_path / "processed"
    rc = rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"])
    assert rc == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    assert len(rows) == 4
    series_dirs = {r["series_dir"] for r in rows}
    assert len(series_dirs) == 2
    counts = {s: sum(1 for r in rows if r["series_dir"] == s) for s in series_dirs}
    assert sorted(counts.values()) == [1, 3]


def test_orders_slices_by_imagepositionpatient(tmp_path: Path, rsna_raw: Path) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    main_series = [
        r for r in rows if r["series_dir"].endswith(
            "1.2.826.0.1.3680043.10.1001_1.2.826.0.1.3680043.10.2001"
        )
    ]
    assert len(main_series) == 3
    assert [int(r["slice_index"]) for r in main_series] == [0, 1, 2]
    series_dir = next(
        d for d in (out / "rsna-ich").iterdir()
        if d.name.endswith("1001_1.2.826.0.1.3680043.10.2001")
    )
    pngs = sorted(series_dir.glob("slice_*.png"))
    assert [p.name for p in pngs] == ["slice_0000.png", "slice_0001.png", "slice_0002.png"]
    assert abs(_read_png_hu(pngs[0]) - (500 * 0.5 - 1024)) < HU_TOL
    assert abs(_read_png_hu(pngs[1]) - (1500 * 0.5 - 1024)) < HU_TOL
    assert abs(_read_png_hu(pngs[2]) - (2500 * 0.5 - 1024)) < HU_TOL


def test_applies_rescale_slope_intercept(tmp_path: Path, rsna_raw: Path) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    first = next(
        r for r in rows
        if r["series_dir"].endswith("1001_1.2.826.0.1.3680043.10.2001")
    )
    hu = _read_png_hu(Path(first["png_path"]))
    assert abs(hu - (500 * 0.5 - 1024)) < HU_TOL


def test_index_carries_hu16_token_and_columns(tmp_path: Path, rsna_raw: Path) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    assert list(rows[0].keys()) == [
        "png_path", "series_dir", "slice_index", "encoding",
        "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
    ]
    assert all(r["encoding"] == hu_encoding_token() for r in rows)
    assert hu_encoding_token() == "hu16_i16_offset32768_clip-1000_4000"
    assert all(r["sequence"] == "ct" for r in rows)


def test_z_spacing_uses_ipp_deltas_over_slice_thickness(
    tmp_path: Path, rsna_raw: Path
) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    main_series = [r for r in rows if r["series_dir"].endswith("1001_1.2.826.0.1.3680043.10.2001")]
    assert all(abs(float(r["spacing_z"]) - 5.0) < 1e-6 for r in main_series)
    assert all(abs(float(r["spacing_x"]) - 0.6) < 1e-6 for r in main_series)
    assert all(abs(float(r["spacing_y"]) - 0.7) < 1e-6 for r in main_series)


def test_patient_id_from_header_with_study_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "a0.dcm", _const(500), study="1.2.3.4", series="1.2.3.5",
              patient="pat-77", ipp_z=0.0)
    _write_ct(raw / "b0.dcm", _const(500), study="1.2.3.9", series="1.2.3.8",
              patient=None, ipp_z=0.0)
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    by_study = {r["series_dir"].split("_")[1]: r["patient_id"] for r in rows}
    assert by_study["1.2.3.4"] == "pat-77"
    assert by_study["1.2.3.9"] == "1.2.3.9"


def test_skips_non_ct_series(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "ct0.dcm", _const(500), study="1.2.3.1", series="1.2.3.2", ipp_z=0.0)
    _write_ct(raw / "mr0.dcm", _const(500), study="1.2.3.3", series="1.2.3.4",
              ipp_z=0.0, modality="MR")
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    assert len(rows) == 1
    assert rows[0]["series_dir"].count("1.2.3.2") == 1


def test_all_non_ct_exits_one(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "mr0.dcm", _const(500), study="1.2.3.3", series="1.2.3.4",
              ipp_z=0.0, modality="MR")
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(raw), "--out-root", str(out), "--min-slices", "1"]) == 1


def test_dedupes_repeated_sop_instance_uid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "s0.dcm", _const(500), study="1.2.3.1", series="1.2.3.2",
              sop="1.2.3.100", ipp_z=0.0)
    _write_ct(raw / "z-s0-copy.dcm", _const(900), study="1.2.3.1", series="1.2.3.2",
              sop="1.2.3.100", ipp_z=0.0)
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    assert len(rows) == 1
    assert abs(_read_png_hu(Path(rows[0]["png_path"])) - (500 * 0.5 - 1024)) < HU_TOL


def test_nested_layout_groups_by_header_uids(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_ct(raw / "dirA" / "s0.dcm", _const(500), study="1.2.3.1", series="1.2.3.2",
              sop="1.2.3.101", ipp_z=0.0)
    _write_ct(raw / "dirB" / "s1.dcm", _const(500), study="1.2.3.1", series="1.2.3.2",
              sop="1.2.3.102", ipp_z=5.0)
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rows = _read_index(out / "rsna-ich" / "index.csv")
    assert len(rows) == 2
    assert rows[0]["series_dir"] == rows[1]["series_dir"]


def test_rerun_resumes_from_meta_without_rewriting_slices(
    tmp_path: Path, rsna_raw: Path
) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    first = _read_index(out / "rsna-ich" / "index.csv")
    png = next((out / "rsna-ich").rglob("slice_0000.png"))
    mtime = png.stat().st_mtime_ns
    meta_file = png.parent / "meta.json"
    meta = json.loads(meta_file.read_text())
    assert meta["encoding"] == hu_encoding_token()
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    second = _read_index(out / "rsna-ich" / "index.csv")
    assert len(second) == len(first)
    assert png.stat().st_mtime_ns == mtime


def test_manifest_out_writes_parquet_with_identity(tmp_path: Path, rsna_raw: Path) -> None:
    out = tmp_path / "processed"
    manifest_path = tmp_path / "manifests" / "rsna_ich.parquet"
    rc = rsna_main([
        "--dicom-dir", str(rsna_raw), "--out-root", str(out),
        "--manifest-out", str(manifest_path), "--min-slices", "1",
    ])
    assert rc == 0
    manifest = DataManifest.load(manifest_path)
    assert len(manifest) == 4
    assert all(r.patient_id for r in manifest.records)
    assert all(r.dataset == "rsna-ich" for r in manifest.records)
    assert all("brain" in r.organs_present for r in manifest.records)
    main_records = [r for r in manifest.records if r.series_id.endswith("1001_1.2.826.0.1.3680043.10.2001")]
    assert all(abs(r.pixel_spacing_x - 0.6) < 1e-6 for r in main_records)
    assert all(abs(r.slice_thickness - 5.0) < 1e-6 for r in main_records)


def test_combine_accepts_ct_and_refuses_mri(
    tmp_path: Path, rsna_raw: Path, monkeypatch
) -> None:
    out = tmp_path / "processed"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(out), "--min-slices", "1"]) == 0
    rsna_index = out / "rsna-ich" / "index.csv"
    mri_index = tmp_path / "mri_index.csv"
    mri_index.write_text(
        "png_path,series_dir,slice_index,encoding,spacing_x,spacing_y,spacing_z,patient_id,sequence\n"
        "x/slice_0000.png,br_x,0,zscore16_u16_offset32768_scale100_clip-300_300,"
        "1.0,1.0,1.0,pat,t1\n"
    )
    ok_out = tmp_path / "combined_ok.csv"
    monkeypatch.setattr(sys, "argv", [
        "mvp_combine_indices.py",
        "--inputs", f"rsna-ich:{rsna_index}",
        "--inputs", f"lidc:{rsna_index}",
        "--out", str(ok_out),
    ])
    assert combine_main() == 0
    bad_out = tmp_path / "combined_bad.csv"
    monkeypatch.setattr(sys, "argv", [
        "mvp_combine_indices.py",
        "--inputs", f"rsna-ich:{rsna_index}",
        "--inputs", f"brats:{mri_index}",
        "--out", str(bad_out),
    ])
    assert combine_main() == 1


def test_registry_entry_uses_canonical_encoding_and_terms_license() -> None:
    registry = DatasetRegistry(_REPO_ROOT / "zoo" / "datasets")
    entry = registry.get("rsna-ich")
    assert entry is not None
    assert entry.modality == "ct"
    assert entry.preprocessing.encoding == "hu16"
    assert entry.preprocessing.index_csv == "index.csv"
    assert "non-commercial" in entry.license.lower()
    assert "CC-BY-SA-4.0" not in entry.license
    assert entry.total_slices == 752803


def test_workers_parallel_matches_serial(tmp_path: Path, rsna_raw: Path) -> None:
    serial_out = tmp_path / "serial"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(serial_out),
                      "--min-slices", "1", "--workers", "1"]) == 0
    par_out = tmp_path / "parallel"
    assert rsna_main(["--dicom-dir", str(rsna_raw), "--out-root", str(par_out),
                      "--min-slices", "1", "--workers", "2"]) == 0
    serial_rows = _read_index(serial_out / "rsna-ich" / "index.csv")
    par_rows = _read_index(par_out / "rsna-ich" / "index.csv")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "png_path"} for r in rows]
    assert strip(par_rows) == strip(serial_rows)
    for r in par_rows:
        assert Path(r["png_path"]).is_file()


def test_main_exit_codes_on_bad_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert rsna_main(["--dicom-dir", str(missing), "--out-root", str(tmp_path / "p")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.md").write_text("nothing here")
    assert rsna_main(["--dicom-dir", str(empty), "--out-root", str(tmp_path / "p")]) == 2

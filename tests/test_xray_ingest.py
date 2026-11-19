"""Tests for X-ray ingestion (scripts/preprocessing/preprocess_xray.py)."""

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

pyd = pytest.importorskip("pydicom")
from PIL import Image

from mvp_combine_indices import main as combine_main
from mvp_prepare_data import index_csv_to_manifest
from preprocess_xray import (
    load_dicom_image,
    load_png_image,
    main as xray_main,
    process_dicom,
)
from zoo.encoding import norm01_encoding_token, u16_to_norm01
from zoo.models import PreprocessingConfig
from zoo.registry import DatasetRegistry

SPACING_X = 0.139
SPACING_Y = 0.194
RESCALE_SLOPE = 0.001
RESCALE_INTERCEPT = -1000.0
NORM_TOL = 2.0 / 65535.0


def _write_dcm(
    path: Path,
    pixels: np.ndarray,
    patient: str = "p02835",
    study: str = "1.2.826.0.1.3680043.2.1125.1.50430799",
    view: str = "AP",
    spacing: tuple[str, str] = ("0.139", "0.194"),
    slope: float = RESCALE_SLOPE,
    intercept: float = RESCALE_INTERCEPT,
) -> None:
    ds = pyd.Dataset()
    ds.SOPClassUID = pyd.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = pyd.uid.generate_uid()
    ds.StudyInstanceUID = study
    ds.PatientID = patient
    ds.ViewPosition = view
    ds.Modality = "CR"
    ds.PatientPosition = "FF"
    ds.RescaleSlope = str(slope)
    ds.RescaleIntercept = str(intercept)
    ds.PixelSpacing = list(spacing)
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


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path)


def _ramp(rows: int = 8, cols: int = 6) -> np.ndarray:
    return (np.arange(rows * cols, dtype=np.uint16) * 80).reshape(rows, cols)


def _read_index(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def test_load_dicom_image_applies_rescale_and_spacing(tmp_path):
    pixels = _ramp()
    p = tmp_path / "a.dcm"
    _write_dcm(p, pixels)
    arr, meta = load_dicom_image(p)
    expected = pixels.astype(np.float32) * RESCALE_SLOPE + RESCALE_INTERCEPT
    np.testing.assert_allclose(arr, expected)
    assert meta["spacing_x"] == pytest.approx(SPACING_X)
    assert meta["spacing_y"] == pytest.approx(SPACING_Y)
    assert meta["patient_id"] == "p02835"
    assert meta["study_id"] == "1.2.826.0.1.3680043.2.1125.1.50430799"
    assert meta["view"] == "ap"


def test_load_dicom_image_falls_back_to_imager_pixel_spacing(tmp_path):
    p = tmp_path / "c.dcm"
    _write_dcm(p, _ramp())
    ds = pyd.dcmread(str(p))
    del ds.PixelSpacing
    ds.ImagerPixelSpacing = ["0.16", "0.16"]
    ds.save_as(str(p), enforce_file_format=True)
    _, meta = load_dicom_image(p)
    assert meta["spacing_x"] == pytest.approx(0.16)
    assert meta["spacing_y"] == pytest.approx(0.16)


def test_load_png_image_normalizes_8bit_to_unit_range(tmp_path):
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    p = tmp_path / "study1_front.png"
    _write_png(p, pixels)
    v, meta = load_png_image(p)
    np.testing.assert_allclose(v, pixels.astype(np.float32) / 255.0, atol=1e-3)
    assert meta["patient_id"] == p.parent.name


def test_process_dicom_writes_norm16_png_and_zero_thickness(tmp_path, capsys):
    pixels = _ramp()
    dcm = tmp_path / "raw" / "p02835" / "s50430799" / "img.dcm"
    _write_dcm(dcm, pixels)
    out_root = tmp_path / "processed"
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])
        n = process_dicom(dcm, out_root, "mimic-cxr", w)
    assert n == 1
    rows = _read_index(index_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["encoding"] == norm01_encoding_token()
    assert row["patient_id"] == "p02835"
    assert row["sequence"] == "ap"
    assert float(row["spacing_x"]) == pytest.approx(SPACING_X)
    assert float(row["spacing_y"]) == pytest.approx(SPACING_Y)
    assert float(row["spacing_z"]) == 0.0
    png_path = Path(row["png_path"])
    assert png_path.exists()
    decoded = u16_to_norm01(np.asarray(Image.open(png_path), dtype=np.uint16))
    rescaled = pixels.astype(np.float32) * RESCALE_SLOPE + RESCALE_INTERCEPT
    vmin, vmax = float(rescaled.min()), float(rescaled.max())
    expected = (rescaled - vmin) / (vmax - vmin)
    np.testing.assert_allclose(decoded, expected, atol=NORM_TOL)


def test_process_dicom_constant_image_encodes_zeros(tmp_path):
    pixels = np.full((6, 6), 1234, dtype=np.uint16)
    dcm = tmp_path / "flat.dcm"
    _write_dcm(dcm, pixels)
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])
        n = process_dicom(dcm, tmp_path / "processed", "padchest", w)
    assert n == 1
    row = _read_index(index_path)[0]
    decoded = np.asarray(Image.open(Path(row["png_path"])), dtype=np.uint16)
    assert int(decoded.max()) == 0


def test_process_dicom_skips_missing_patient_id(tmp_path, capsys):
    dcm = tmp_path / "anon.dcm"
    _write_dcm(dcm, _ramp(), patient="")
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        n = process_dicom(dcm, tmp_path / "processed", "mimic-cxr", w)
    assert n == 0
    out = capsys.readouterr().out
    assert "skip=true" in out
    assert "no_patient_id" in out
    assert _read_index(index_path) == []


def test_process_dicom_groups_views_under_one_series(tmp_path):
    base = tmp_path / "raw"
    _write_dcm(base / "p1" / "s1" / "a.dcm", _ramp(), patient="p1",
                  study="1.2.826.0.1.3680043.2.1125.1.50430009", view="AP")
    _write_dcm(base / "p1" / "s1" / "b.dcm", _ramp(), patient="p1",
                  study="1.2.826.0.1.3680043.2.1125.1.50430009", view="PA")
    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])
        for name in ("a.dcm", "b.dcm"):
            n = process_dicom(base / "p1" / "s1" / name, tmp_path / "processed", "x", w)
            assert n == 1
    rows = _read_index(index_path)
    series = {r["series_dir"] for r in rows}
    assert len(series) == 1
    assert sorted(int(r["slice_index"]) for r in rows) == [0, 1]
    assert sorted(r["sequence"] for r in rows) == ["ap", "pa"]


def test_process_dicom_resume_reindexes_existing_series(tmp_path):
    dcm = tmp_path / "img.dcm"
    _write_dcm(dcm, _ramp())
    out_root = tmp_path / "processed"
    first = tmp_path / "first.csv"
    with first.open("w", newline="") as f:
        process_dicom(dcm, out_root, "x", csv.writer(f), {})
    pngs = list((out_root / "x").rglob("slice_*.png"))
    assert len(pngs) == 1

    second = tmp_path / "second.csv"
    with second.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])
        n = process_dicom(dcm, out_root, "x", w)
    assert n == 1
    rows = _read_index(second)
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "p02835"
    assert len(list((out_root / "x").rglob("slice_*.png"))) == 1


def test_xray_main_dicom_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw"
    _write_dcm(raw / "p00001" / "st1" / "img1.dcm", _ramp(), patient="p00001",
                  study="1.2.826.0.1.3680043.2.1125.1.50430001")
    _write_dcm(raw / "p00002" / "st2" / "img2.dcm", _ramp(), patient="p00002",
                  study="1.2.826.0.1.3680043.2.1125.1.50430002", view="PA")
    out_root = tmp_path / "processed"
    rc = xray_main([
        "--dicom-dir", str(raw),
        "--out-root", str(out_root),
        "--dataset-name", "mimic-cxr",
    ])
    assert rc == 0
    rows = _read_index(out_root / "mimic-cxr" / "index.csv")
    assert len(rows) == 2
    assert {r["patient_id"] for r in rows} == {"p00001", "p00002"}
    assert {r["sequence"] for r in rows} == {"ap", "pa"}
    assert all(float(r["spacing_z"]) == 0.0 for r in rows)
    assert all(r["encoding"] == norm01_encoding_token() for r in rows)


def test_xray_main_png_end_to_end(tmp_path):
    pngs = tmp_path / "png"
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    _write_png(pngs / "train" / "Patient00001" / "study1_front.png", img)
    _write_png(pngs / "train" / "Patient00001" / "study1_left.png", img)
    out_root = tmp_path / "processed"
    rc = xray_main([
        "--png-dir", str(pngs),
        "--out-root", str(out_root),
        "--dataset-name", "chexpert",
    ])
    assert rc == 0
    rows = _read_index(out_root / "chexpert" / "index.csv")
    assert len(rows) == 2
    assert {r["patient_id"] for r in rows} == {"Patient00001"}
    assert {r["sequence"] for r in rows} == {"front", "left"}
    assert all(r["series_dir"].endswith("_Patient00001_study1") for r in rows)
    decoded = u16_to_norm01(np.asarray(Image.open(Path(rows[0]["png_path"])), dtype=np.uint16))
    v = img.astype(np.float32) / 255.0
    expected = (v - v.min()) / (v.max() - v.min())
    np.testing.assert_allclose(decoded, expected, atol=NORM_TOL)


def test_xray_main_missing_input_dir_exits_2(tmp_path):
    rc = xray_main([
        "--dicom-dir", str(tmp_path / "nope"),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "x",
    ])
    assert rc == 2


def test_xray_main_no_files_exits_2(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rc = xray_main([
        "--dicom-dir", str(raw),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "x",
    ])
    assert rc == 2


def test_xray_main_all_skipped_exits_1(tmp_path, capsys):
    raw = tmp_path / "raw"
    _write_dcm(raw / "anon" / "s" / "img.dcm", _ramp(), patient="")
    rc = xray_main([
        "--dicom-dir", str(raw),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "x",
    ])
    assert rc == 1


def _write_xray_index(path: Path, dataset: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])
        w.writerow(["x.png", f"{dataset}_p1_s1", 0, norm01_encoding_token(),
                    "0.139", "0.194", "0.0", "p1", "ap"])


def _write_ct_index(path: Path, dataset: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z"])
        w.writerow(["y.png", f"{dataset}_s1", 0,
                    "hu16_i16_offset32768_clip-1000_4000", "0.7", "0.7", "1.5"])


def test_combine_refuses_xray_ct_mix(tmp_path, monkeypatch, capsys):
    xray_csv = tmp_path / "xray_index.csv"
    ct_csv = tmp_path / "ct_index.csv"
    _write_xray_index(xray_csv, "mimic-cxr")
    _write_ct_index(ct_csv, "lidc-idri")
    out = tmp_path / "combined" / "index.csv"
    monkeypatch.setattr(sys, "argv", [
        "combine_indices.py",
        "--inputs", f"mimic-cxr:{xray_csv}",
        "--inputs", f"lidc-idri:{ct_csv}",
        "--out", str(out),
    ])
    assert combine_main() != 0
    err = capsys.readouterr().err
    assert "norm16" in err
    assert not out.exists()


def test_combine_passes_xray_columns(tmp_path, monkeypatch):
    a = tmp_path / "a_index.csv"
    b = tmp_path / "b_index.csv"
    _write_xray_index(a, "mimic-cxr")
    _write_xray_index(b, "padchest")
    out = tmp_path / "combined" / "index.csv"
    monkeypatch.setattr(sys, "argv", [
        "combine_indices.py",
        "--inputs", f"mimic-cxr:{a}",
        "--inputs", f"padchest:{b}",
        "--out", str(out),
    ])
    assert combine_main() == 0
    rows = _read_index(out)
    assert {r["patient_id"] for r in rows} == {"p1"}
    assert {r["sequence"] for r in rows} == {"ap"}


def test_index_csv_to_manifest_keeps_xray_identity_and_zero_thickness(tmp_path):
    dcm = tmp_path / "raw" / "p1" / "st1" / "img.dcm"
    _write_dcm(dcm, _ramp(), patient="p1")
    rc = xray_main([
        "--dicom-dir", str(tmp_path / "raw"),
        "--out-root", str(tmp_path / "processed"),
        "--dataset-name", "mimic-cxr",
    ])
    assert rc == 0
    manifest = index_csv_to_manifest(
        tmp_path / "processed" / "mimic-cxr" / "index.csv", "mimic-cxr", ["chest"]
    )
    record = manifest.records[0]
    assert record.patient_id == "p1"
    assert record.slice_thickness == 0.0


def test_preprocessing_config_accepts_norm16():
    assert PreprocessingConfig(encoding="norm16").encoding == "norm16"


def test_real_catalog_has_xray_entries():
    registry = DatasetRegistry(_REPO_ROOT / "zoo" / "datasets")
    xray = registry.query(modality="xray")
    names = {e.name for e in xray}
    assert {"chexpert", "mimic-cxr", "padchest"} <= names
    for entry in xray:
        assert entry.license
        assert entry.citation
        assert entry.total_slices > 0
        assert entry.preprocessing.encoding == "norm16"

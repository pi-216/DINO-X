"""Tests for Mayo LDCT ingestion (scripts/preprocessing/preprocess_mayo_ldct.py)."""

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

from mvp_combine_indices import main as combine_main
from mvp_prepare_data import index_csv_to_manifest
from preprocess_mayo_ldct import main as mayo_main
from zoo.encoding import hu_encoding_token, u16_to_hu
from zoo.manifest import DataManifest
from zoo.registry import DatasetRegistry

HU_TOL = 0.05
RAW_DATA_STORAGE = "1.2.840.10008.5.1.4.1.1.66"

STUDY_CHEST = "1.2.840.113713.4.100.1.2.9001"
SERIES_CHEST = "1.2.840.113713.4.100.1.3.9001"
SERIES_CHEST_PROJ = "1.2.840.113713.4.100.1.3.9002"
STUDY_ABD = "1.2.840.113713.4.100.1.2.9010"
SERIES_ABD = "1.2.840.113713.4.100.1.3.9010"


def _write_ct(
    path: Path,
    stored: np.ndarray,
    *,
    study: str,
    series: str,
    patient: str,
    ipp_z: float = 0.0,
    instance: int = 1,
    spacing: tuple[str, str] = ("0.736328", "0.736328"),
    slice_thickness: str = "0.625",
    spacing_between: str | None = "0.625",
    slope: float = 1.0,
    intercept: float = -1024.0,
    body_part: str = "CHEST",
    series_description: str = "Full Dose Images",
) -> None:
    ds = pyd.Dataset()
    ds.SOPClassUID = pyd.uid.CTImageStorage
    ds.SOPInstanceUID = pyd.uid.generate_uid()
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.Modality = "CT"
    ds.PatientID = patient
    ds.BodyPartExamined = body_part
    ds.SeriesDescription = series_description
    ds.InstanceNumber = str(instance)
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


def _write_projection(
    path: Path,
    *,
    study: str,
    series: str,
    patient: str,
    instance: int = 1,
) -> None:
    ds = pyd.Dataset()
    ds.SOPClassUID = RAW_DATA_STORAGE
    ds.SOPInstanceUID = pyd.uid.generate_uid()
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.Modality = "CT"
    ds.PatientID = patient
    ds.BodyPartExamined = "CHEST"
    ds.SeriesDescription = "Low Dose Projections"
    ds.InstanceNumber = str(instance)
    ds.Rows = 888
    ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((888, 64), dtype="<u2").tobytes()
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


def _read_png_u16(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path))


def _read_png_hu(path: Path) -> float:
    return float(u16_to_hu(_read_png_u16(path))[0][0])


@pytest.fixture()
def mayo_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_ct(
            raw / SERIES_CHEST / f"a{i}.dcm", _const(24 + i),
            study=STUDY_CHEST, series=SERIES_CHEST, patient="C001",
            ipp_z=0.625 * i, instance=i + 1,
        )
    for i in range(3):
        _write_projection(
            raw / SERIES_CHEST_PROJ / f"p{i}.dcm",
            study=STUDY_CHEST, series=SERIES_CHEST_PROJ, patient="C001",
            instance=i + 1,
        )
    for i in range(3):
        _write_ct(
            raw / SERIES_ABD / f"b{i}.dcm", _const(2024),
            study=STUDY_ABD, series=SERIES_ABD, patient="L001",
            ipp_z=5.0 * i, instance=i + 1, body_part="ABDOMEN",
            slice_thickness="5.0", spacing=("0.9765625", "0.9765625"),
        )
    return raw


def test_projection_series_never_reach_the_index(
    tmp_path: Path, mayo_raw: Path,
) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "mayo-ldct" / "index.csv")
    assert all(r["encoding"] == hu_encoding_token() for r in rows)
    assert {r["series_dir"] for r in rows} == {
        f"mayo-ldct_{STUDY_CHEST}_{SERIES_CHEST}",
        f"mayo-ldct_{STUDY_ABD}_{SERIES_ABD}",
    }
    assert len(rows) == 6
    assert not (out / "mayo-ldct" / f"mayo-ldct_{STUDY_CHEST}_{SERIES_CHEST_PROJ}").exists()


def test_index_columns_carry_per_row_organs(tmp_path: Path, mayo_raw: Path) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    with (out / "mayo-ldct" / "index.csv").open("r", newline="") as f:
        header = csv.reader(f).__next__()
    assert header == [
        "png_path", "series_dir", "slice_index", "encoding",
        "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
        "organs",
    ]


def test_hu16_canonical_scale_on_disk(tmp_path: Path, mayo_raw: Path) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "mayo-ldct" / "index.csv")
    chest_rows = [r for r in rows if r["series_dir"] == f"mayo-ldct_{STUDY_CHEST}_{SERIES_CHEST}"]
    chest_rows.sort(key=lambda r: int(r["slice_index"]))
    assert [_read_png_hu(Path(r["png_path"])) for r in chest_rows] == pytest.approx(
        [-1000.0, -999.0, -998.0], abs=HU_TOL
    )
    abd_row = next(r for r in rows if r["series_dir"] == f"mayo-ldct_{STUDY_ABD}_{SERIES_ABD}")
    assert int(_read_png_u16(Path(abd_row["png_path"]))[0][0]) == 42768


def test_organs_come_from_body_part_examined(tmp_path: Path, mayo_raw: Path) -> None:
    out = tmp_path / "processed"
    manifest_path = tmp_path / "mayo.parquet"
    assert mayo_main([
        "--dicom-dir", str(mayo_raw), "--out-root", str(out),
        "--manifest-out", str(manifest_path),
    ]) == 0
    manifest = DataManifest.load(manifest_path)
    by_patient = {}
    for rec in manifest.records:
        by_patient.setdefault(rec.patient_id, set()).update(rec.organs_present)
    assert by_patient["C001"] == {"chest", "lung"}
    assert by_patient["L001"] == {"abdomen", "liver"}
    assert len(manifest) == 6


def test_spacing_columns_record_measured_values(tmp_path: Path, mayo_raw: Path) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    rows = _read_index(out / "mayo-ldct" / "index.csv")
    chest = [r for r in rows if r["series_dir"].endswith(SERIES_CHEST)]
    assert all(abs(float(r["spacing_x"]) - 0.736328) < 1e-6 for r in chest)
    assert all(abs(float(r["spacing_z"]) - 0.625) < 1e-6 for r in chest)
    abd = [r for r in rows if r["series_dir"].endswith(SERIES_ABD)]
    assert all(abs(float(r["spacing_x"]) - 0.9765625) < 1e-6 for r in abd)


def test_rerun_reindexes_without_rewriting_slices(tmp_path: Path, mayo_raw: Path) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    first = _read_index(out / "mayo-ldct" / "index.csv")
    mtimes = {
        p: p.stat().st_mtime_ns
        for p in (out / "mayo-ldct").rglob("slice_*.png")
    }
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    second = _read_index(out / "mayo-ldct" / "index.csv")
    assert second == first
    assert {
        p: p.stat().st_mtime_ns
        for p in (out / "mayo-ldct").rglob("slice_*.png")
    } == mtimes


def test_registry_entry_uses_canonical_encoding() -> None:
    registry = DatasetRegistry(_REPO_ROOT / "zoo" / "datasets")
    entry = registry.get("mayo-ldct")
    assert entry is not None
    assert entry.modality == "ct"
    assert entry.preprocessing.encoding == "hu16"
    assert entry.preprocessing.index_csv == "index.csv"
    assert entry.total_slices > 0
    assert entry.total_series > 0
    raw = (_REPO_ROOT / "zoo" / "datasets" / "ct" / "mayo_ldct.yaml").read_text()
    block = raw.split("preprocessing:", 1)[1].split("citation:", 1)[0]
    assert "encoding: hu16" in block
    assert "hu_shift" not in block
    assert "scale:" not in block


def test_combined_index_accepts_mayo_rows(tmp_path: Path, mayo_raw: Path,
                                          monkeypatch) -> None:
    out = tmp_path / "processed"
    assert mayo_main(["--dicom-dir", str(mayo_raw), "--out-root", str(out)]) == 0
    combined = tmp_path / "combined.csv"
    monkeypatch.setattr(sys, "argv", [
        "mvp_combine_indices.py",
        "--inputs", f"mayo-ldct:{out / 'mayo-ldct' / 'index.csv'}",
        "--inputs", f"mayo-ldct2:{out / 'mayo-ldct' / 'index.csv'}",
        "--out", str(combined),
    ])
    rc = combine_main()
    assert rc == 0
    rows = _read_index(combined)
    assert len(rows) == 12
    assert all(r["patient_id"] in {"C001", "L001"} for r in rows)


def test_index_to_manifest_prefers_row_organs(tmp_path: Path) -> None:
    csv_path = tmp_path / "index.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "png_path", "series_dir", "slice_index", "encoding",
            "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
            "organs",
        ])
        w.writerow([
            "x.png", "s1", 0, "hu16", "0.7", "0.7", "1.0", "C001", "ct",
            "chest;lung",
        ])
        w.writerow([
            "y.png", "s1", 1, "hu16", "0.7", "0.7", "1.0", "C001", "ct", "",
        ])
    manifest = index_csv_to_manifest(csv_path, "mayo-ldct", ["global-organ"])
    recs = manifest.records
    assert recs[0].organs_present == ["chest", "lung"]
    assert recs[1].organs_present == ["global-organ"]

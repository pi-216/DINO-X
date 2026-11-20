#!/usr/bin/env python3
"""CQ500 head CT DICOM -> canonical hu16 PNG slices.

CQ500 (Qure.ai, 491 head CT studies) ships as a nested tree:
`raw/qct<NN>/CQ500CT<N> CQ500CT<N>/<StudyDescription>/<SeriesDescription>/`
of `CT######.dcm` files. Series are grouped from the DICOM headers —
`(StudyInstanceUID, SeriesInstanceUID)` — not from the directory names, so
duplicate protocol directories remain distinct series.

Handling:
  - Only series with Modality CT are ingested; other series are skipped.
  - SOPInstanceUID duplicates are dropped (first file wins) so a rerun or a
    double-shipped slice never produces a doubled stack.
  - Slices are ordered by ImagePositionPatient z (fallback InstanceNumber);
    z-spacing is the median adjacent-slice IPP delta, falling back to
    SpacingBetweenSlices then SliceThickness. CQ500 thick-slice series
    repeat a constant SliceThickness that disagrees with the acquisition
    deltas (e.g. 5.0 mm slices shipped 10.125 mm apart).
  - HU = stored * RescaleSlope + RescaleIntercept, stored with the canonical
    `hu16_png` encoding from zoo.encoding (never inlined here).
  - patient_id comes from the DICOM PatientID (`CQ500-CT-<N>`) and falls
    back to the StudyInstanceUID, so zoo.splits patient-level leakage
    asserts can be enforced downstream.
  - Each series directory carries meta.json; a rerun without --force
    re-indexes from it without rewriting slice PNGs.
  - --manifest-out writes the Parquet manifest (per-slice spacing +
    identity) alongside the index CSV.

The 2026-04 corpus under `data/cq500/processed-hu16/` predates the canonical
hu16 scale fix and is NOT consumable; this script writes to
`data/processed/cq500/` instead.

Usage:
    python scripts/preprocessing/preprocess_cq500.py \
        --dicom-dir data/cq500/raw \
        --out-root data/processed \
        --manifest-out data/manifests/cq500.parquet
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
for _p in (_REPO_ROOT, str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zoo.encoding import hu_encoding_token, hu_to_u16

SKIP_SUFFIXES = {
    ".zip", ".csv", ".md", ".txt", ".png", ".jpg", ".json", ".gz",
    ".tar", ".parquet", ".complete", ".sha256",
}
INDEX_COLUMNS = [
    "png_path", "series_dir", "slice_index", "encoding",
    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence",
]
CT_SEQUENCE = "ct"


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. pip install {mod}")


try:
    import pydicom  # type: ignore
    from pydicom.dataset import FileMetaDataset  # type: ignore
    from pydicom.uid import ImplicitVRLittleEndian  # type: ignore
except ImportError:
    _need("pydicom")


@dataclass(frozen=True)
class SliceEntry:
    path: Path
    study: str
    series: str
    sop: str
    modality: str
    instance: int
    sort_z: float
    has_ipp: bool
    patient_id: str
    spacing_x: float | None
    spacing_y: float | None
    slice_thickness: float | None
    spacing_between: float | None


def iter_dicom_files(dicom_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, fnames in os.walk(dicom_dir):
        for fname in fnames:
            fp = Path(root) / fname
            if fp.suffix.lower() in SKIP_SUFFIXES or fname == "LICENSE":
                continue
            if fp.suffix and fp.suffix.lower() != ".dcm":
                continue
            files.append(fp)
    files.sort(key=str)
    return files


def read_entry(path: Path) -> SliceEntry | None:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
        return None
    study = str(getattr(ds, "StudyInstanceUID", "") or "")
    series = str(getattr(ds, "SeriesInstanceUID", "") or "")
    sop = str(getattr(ds, "SOPInstanceUID", "") or "")
    if not study or not series:
        return None
    ipp = getattr(ds, "ImagePositionPatient", None)
    instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    has_ipp = ipp is not None and len(ipp) >= 3
    sort_z = float(ipp[2]) if has_ipp else float(instance)
    patient = str(getattr(ds, "PatientID", "") or "") or study
    ps = getattr(ds, "PixelSpacing", None)
    spacing_x = float(ps[0]) if ps is not None and len(ps) >= 2 else None
    spacing_y = float(ps[1]) if ps is not None and len(ps) >= 2 else None

    def _tag(name: str) -> float | None:
        val = getattr(ds, name, None)
        return float(val) if val is not None else None

    return SliceEntry(
        path=path, study=study, series=series, sop=sop,
        modality=str(getattr(ds, "Modality", "") or "").upper(),
        instance=instance, sort_z=sort_z, has_ipp=has_ipp,
        patient_id=patient,
        spacing_x=spacing_x, spacing_y=spacing_y,
        slice_thickness=_tag("SliceThickness"),
        spacing_between=_tag("SpacingBetweenSlices"),
    )


def group_series(files: list[Path]) -> tuple[dict[tuple[str, str], list[SliceEntry]], int]:
    groups: dict[tuple[str, str], list[SliceEntry]] = {}
    unreadable = 0
    for fp in files:
        entry = read_entry(fp)
        if entry is None:
            unreadable += 1
            continue
        groups.setdefault((entry.study, entry.series), []).append(entry)
    return groups, unreadable


def series_spacing(entries: list[SliceEntry]) -> tuple[float, float, float]:
    spacing_x = next((e.spacing_x for e in entries if e.spacing_x is not None), 1.0)
    spacing_y = next((e.spacing_y for e in entries if e.spacing_y is not None), spacing_x)
    ipp_entries = [e for e in entries if e.has_ipp]
    if len(ipp_entries) >= 2:
        positions = sorted({e.sort_z for e in ipp_entries})
        deltas = sorted(
            abs(positions[i + 1] - positions[i])
            for i in range(len(positions) - 1)
        )
        spacing_z = deltas[len(deltas) // 2]
        if spacing_z > 0:
            return spacing_x, spacing_y, spacing_z
    between = next((e.spacing_between for e in entries if e.spacing_between), None)
    thickness = next((e.slice_thickness for e in entries if e.slice_thickness), None)
    return spacing_x, spacing_y, between or thickness or 1.0


def load_hu_slice(path: Path) -> "np.ndarray":
    ds = pydicom.dcmread(str(path), force=True)
    if getattr(ds, "file_meta", None) is None:
        ds.file_meta = FileMetaDataset()
    if getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def save_slice_hu16(hu_slice: "np.ndarray", out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hu_to_u16(hu_slice)).save(out_path, format="PNG")


def _meta_path(series_dir: Path) -> Path:
    return series_dir / "meta.json"


def _row_for(
    png_path: Path,
    series_name: str,
    slice_idx: int,
    spacing: tuple[float, float, float],
    patient_id: str,
) -> list[str]:
    return [
        str(png_path),
        series_name,
        str(slice_idx),
        hu_encoding_token(),
        f"{spacing[0]:.6f}",
        f"{spacing[1]:.6f}",
        f"{spacing[2]:.6f}",
        patient_id,
        CT_SEQUENCE,
    ]


def _reindex_existing(
    series_dir: Path,
    series_name: str,
) -> tuple[list[list[str]], tuple[float, float, float] | None]:
    meta_file = _meta_path(series_dir)
    if not meta_file.exists():
        return [], None
    meta = json.loads(meta_file.read_text())
    spacing = tuple(meta["spacing"])
    rows: list[list[str]] = []
    for p in sorted(series_dir.glob("slice_*.png")):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except ValueError:
            continue
        rows.append(_row_for(
            p, series_name, idx, spacing, meta["patient_id"],
        ))
    return rows, spacing


def ingest_group(
    key: tuple[str, str],
    entries: list[SliceEntry],
    out_root: Path,
    dataset_name: str,
    min_slices: int = 3,
    force: bool = False,
) -> tuple[list[list[str]], int, tuple[float, float, float] | None]:
    study, series = key
    seen_sop: set[str] = set()
    unique: list[SliceEntry] = []
    for e in sorted(entries, key=lambda e: (e.sort_z, e.instance, str(e.path))):
        if e.sop and e.sop in seen_sop:
            continue
        if e.sop:
            seen_sop.add(e.sop)
        unique.append(e)

    if unique and unique[0].modality != "CT":
        print(f"skip=true series={series} reason=non_ct_modality "
              f"modality={unique[0].modality} files={len(unique)}", flush=True)
        return [], 0, None

    series_name = f"{dataset_name}_{study}_{series}"
    out_series_dir = out_root / dataset_name / series_name

    if not force and out_series_dir.exists():
        rows, spacing = _reindex_existing(out_series_dir, series_name)
        if rows:
            print(f"skip=true series={series} reason=already_converted "
                  f"slices={len(rows)}")
            return rows, len(rows), spacing

    if len(unique) < min_slices:
        print(f"skip=true series={series} reason=too_few_slices "
              f"slices={len(unique)}")
        return [], 0, None

    spacing = series_spacing(unique)
    patient_id = unique[0].patient_id
    rows: list[list[str]] = []
    base_shape: tuple[int, int] | None = None

    for e in unique:
        try:
            hu = load_hu_slice(e.path)
        except Exception as exc:
            print(f"warn=true series={series} file={e.path.name} "
                  f"reason=pixel_load_failed err={exc}", flush=True)
            continue
        shape = (int(hu.shape[0]), int(hu.shape[1]))
        if base_shape is None:
            base_shape = shape
        elif base_shape != shape:
            continue
        out_path = out_series_dir / f"slice_{len(rows):04d}.png"
        save_slice_hu16(hu, out_path)
        rows.append(_row_for(out_path, series_name, len(rows), spacing, patient_id))

    n_slices = len(rows)
    if n_slices == 0:
        print(f"skip=true series={series} reason=no_decodable_slices")
        return [], 0, None

    _meta_path(out_series_dir).write_text(json.dumps({
        "spacing": list(spacing),
        "patient_id": patient_id,
        "study_uid": study,
        "series_uid": series,
        "n_slices": n_slices,
        "sequence": CT_SEQUENCE,
        "encoding": hu_encoding_token(),
    }))
    print(f"ok=true series={series} slices={n_slices} "
          f"spacing=({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f})", flush=True)
    return rows, n_slices, spacing


def _ingest_job(job: tuple) -> tuple:
    return ingest_group(*job)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Convert CQ500 head CT DICOM to canonical hu16 16-bit PNG "
            "slices with a spacing/identity index CSV."
        ),
    )
    ap.add_argument("--dicom-dir", type=Path, required=True,
                    help="Directory containing raw CQ500 DICOM files")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"),
                    help="Root output directory for processed data")
    ap.add_argument("--dataset-name", type=str, default="cq500",
                    help="Dataset identifier")
    ap.add_argument("--min-slices", type=int, default=3,
                    help="Skip series with fewer DICOM slices than this")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="Also write a Parquet manifest at this path")
    ap.add_argument("--organs", type=str, default="brain,cranial",
                    help="Comma-separated organs recorded in the manifest")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess series even if output already exists")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel processes for series conversion (1 = serial)")
    args = ap.parse_args(argv)

    dicom_dir = args.dicom_dir
    if not dicom_dir.is_dir():
        print(f"ERROR: DICOM directory not found: {dicom_dir}", file=sys.stderr)
        return 2

    files = iter_dicom_files(dicom_dir)
    if not files:
        print(f"ERROR: no DICOM files found under: {dicom_dir}", file=sys.stderr)
        return 2

    groups, unreadable = group_series(files)
    if unreadable:
        print(f"warning: {unreadable} unreadable file(s) skipped", file=sys.stderr)
    print(f"Found {len(files)} candidate files in {len(groups)} series")

    index_path = args.out_root / args.dataset_name / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    total_series = 0
    spacings: list[tuple[float, float, float]] = []

    keys = sorted(groups)
    jobs = [(k, groups[k], args.out_root, args.dataset_name,
             args.min_slices, args.force) for k in keys]

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLUMNS)
        if args.workers == 1:
            results = [ingest_group(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=mp.get_context("forkserver"),
                ) as pool:
                results = list(pool.map(_ingest_job, jobs))
        for rows, n, spacing in results:
            for row in rows:
                w.writerow(row)
            if n > 0:
                total_slices += n
                total_series += 1
                if spacing is not None:
                    spacings.append(spacing)

    print(f"\nDone: {total_series} series, {total_slices} slices")
    print(f"Index: {index_path}")

    if spacings:
        for axis, name in zip((0, 1, 2), ("pixel_x", "pixel_y", "z")):
            vals = [s[axis] for s in spacings]
            print(f"  {name}: min={min(vals):.4f} max={max(vals):.4f}")

    if total_slices == 0:
        print("ERROR: no slices ingested — check --dicom-dir and Modality CT",
              file=sys.stderr)
        return 1

    if args.manifest_out is not None:
        from mvp_prepare_data import index_csv_to_manifest

        organs = [o.strip() for o in args.organs.split(",") if o.strip()]
        manifest = index_csv_to_manifest(index_path, args.dataset_name, organs)
        manifest.save(args.manifest_out)
        print(f"Manifest: {args.manifest_out} ({len(manifest)} slices)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

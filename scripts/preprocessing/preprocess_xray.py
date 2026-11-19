#!/usr/bin/env python3
"""DICOM / PNG radiographs -> [0,1]-normalized 16-bit PNG slices.

Ingests 2D radiograph datasets (MIMIC-CXR, PadChest, CheXpert Plus as
DICOM; CheXpert as PNG) into the same slice-PNG + index-CSV layout as the
CT and MRI pipelines, with radiograph-specific handling:

  - Images are 2D: index rows carry spacing_z (slice thickness) 0.0 so the
    ScaleEmbedding consumes an explicit planar thickness.
  - DICOM in-plane spacing comes from PixelSpacing (0028,0030), falling
    back to ImagerPixelSpacing (0018,1164), then 1.0. PNG inputs have no
    spacing metadata and default to 1.0.
  - Radiograph intensity is optical density with display-dependent
    rescaling, so each image has RescaleSlope/RescaleIntercept applied and
    is min-max normalized to [0, 1], then stored with the canonical
    `norm16_png` encoding from zoo.encoding (never inlined here). 8-bit
    PNGs are stored as value/255; 16-bit PNGs as value/65535.
  - Identity: DICOM rows take patient_id from PatientID (0010,0020) and
    group views under StudyInstanceUID (0020,000D); an empty PatientID
    skips the image so patient-level split leakage asserts cannot silently
    collapse distinct patients into one. PNG mode (CheXpert layout
    Patient/study_view.png) takes the patient from the parent directory
    and the study/view from the filename.
  - The view (DICOM ViewPosition (0018,5101), or the filename token
    front/left/right/ap/pa/lateral/oblique) is written to the `sequence`
    column so the combined-corpus index schema stays uniform.
  - Each series directory carries meta.json (spacing, patient, per-slice
    view and min/max stats, processed source names); a rerun without
    --force re-indexes already-ingested sources from it.

Usage:
    python scripts/preprocessing/preprocess_xray.py \
        --dicom-dir data/raw/mimic-cxr/files \
        --out-root data/processed \
        --dataset-name mimic-cxr

    python scripts/preprocessing/preprocess_xray.py \
        --png-dir data/raw/chexpert/images \
        --out-root data/processed \
        --dataset-name chexpert
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zoo.encoding import norm01_encoding_token, norm01_to_u16

VIEW_TOKENS = ("front", "left", "right", "ap", "pa", "lateral", "oblique")
VIEW_TOKEN_RE = re.compile(
    r"^(.*)_(" + "|".join(VIEW_TOKENS) + r")$", re.IGNORECASE
)
SLICE_THICKNESS_ZERO = "0.000000"


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. pip install {mod}")


def _series_meta_path(series_dir: Path) -> Path:
    return series_dir / "meta.json"


def _spacing_from_dicom(ds) -> tuple[float, float]:
    raw = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    if raw is None:
        return 1.0, 1.0
    try:
        sx, sy = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return 1.0, 1.0
    return (sx if sx > 0 else 1.0, sy if sy > 0 else 1.0)


def load_dicom_image(path: Path) -> tuple[np.ndarray, dict]:
    """Read one DICOM radiograph; apply the modality LUT, return (floats, meta)."""
    try:
        import pydicom
    except ImportError:
        _need("pydicom")
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-frame 2D pixels, got shape {arr.shape}")
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    if slope is not None:
        arr = arr * float(slope)
    if intercept is not None:
        arr = arr + float(intercept)
    spacing_x, spacing_y = _spacing_from_dicom(ds)
    meta = {
        "patient_id": str(getattr(ds, "PatientID", "") or "").strip(),
        "study_id": str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        or path.parent.name,
        "view": str(getattr(ds, "ViewPosition", "") or "").strip().lower()
        or "unknown",
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
    }
    return arr, meta


def load_png_image(path: Path) -> tuple[np.ndarray, dict]:
    """Read one radiograph PNG, return (floats in [0,1], meta)."""
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")
    if img.mode == "L":
        v = np.asarray(img, dtype=np.float32) / 255.0
    elif img.mode in ("I;16", "I;16B", "I;16L", "I"):
        v = np.asarray(img, dtype=np.float32) / 65535.0
    else:
        raise ValueError(f"Unsupported PNG mode: {img.mode}")
    stem = path.stem
    m = VIEW_TOKEN_RE.match(stem)
    base, view = (m.group(1), m.group(2).lower()) if m else (stem, "unknown")
    meta = {
        "patient_id": path.parent.name,
        "study_id": base,
        "view": view,
        "spacing_x": 1.0,
        "spacing_y": 1.0,
    }
    return v, meta


def normalize_minmax(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-image min-max to [0,1]; constant images -> zeros."""
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin <= 1e-8:
        return np.zeros_like(arr), vmin, vmax
    return (arr - vmin) / (vmax - vmin), vmin, vmax


def _row_for(
    png_path: Path,
    series_name: str,
    slice_idx: int,
    spacing: tuple[float, float],
    patient_id: str,
    view: str,
) -> list[str]:
    return [
        str(png_path),
        series_name,
        str(slice_idx),
        norm01_encoding_token(),
        f"{spacing[0]:.6f}",
        f"{spacing[1]:.6f}",
        SLICE_THICKNESS_ZERO,
        patient_id,
        view,
    ]


def _load_meta(series_dir: Path, spacing: tuple[float, float], patient_id: str) -> dict:
    meta_file = _series_meta_path(series_dir)
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            if isinstance(meta, dict) and "images" in meta:
                return meta
        except json.JSONDecodeError:
            pass
    return {
        "spacing": [spacing[0], spacing[1], 0.0],
        "patient_id": patient_id,
        "images": {},
        "sources": [],
        "encoding": norm01_encoding_token(),
    }


def _reindex_series(
    series_dir: Path,
    series_name: str,
    meta: dict,
    writer: csv.writer,
) -> int:
    spacing = tuple(meta.get("spacing", [1.0, 1.0, 0.0]))[:2]
    n = 0
    for idx in sorted(int(k) for k in meta.get("images", {})):
        out_path = series_dir / f"slice_{idx:04d}.png"
        if not out_path.exists():
            continue
        entry = meta["images"][str(idx)]
        writer.writerow(_row_for(
            out_path, series_name, idx, spacing,
            meta["patient_id"], entry.get("view", "unknown"),
        ))
        n += 1
    return n


def _process_image(
    intensity: np.ndarray,
    meta_src: dict,
    source_key: str,
    series_name: str,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
    file_label: str,
    force: bool = False,
) -> int:
    spacing = (meta_src["spacing_x"], meta_src["spacing_y"])
    patient_id = meta_src["patient_id"]
    out_series_dir = out_root / dataset_name / series_name

    meta = _load_meta(out_series_dir, spacing, patient_id)
    if not force and source_key in meta.get("sources", []):
        n = _reindex_series(out_series_dir, series_name, meta, writer)
        if n > 0:
            print(f"skip=true file={file_label} reason=already_converted slices={n}")
            return n

    v, vmin, vmax = normalize_minmax(intensity)
    idx = max((int(k) for k in meta.get("images", {})), default=-1) + 1
    out_path = out_series_dir / f"slice_{idx:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(norm01_to_u16(v)).save(out_path, format="PNG")

    meta["spacing"] = [spacing[0], spacing[1], 0.0]
    meta["patient_id"] = patient_id
    meta["images"][str(idx)] = {
        "view": meta_src["view"], "min": vmin, "max": vmax,
    }
    meta.setdefault("sources", []).append(source_key)
    meta["encoding"] = norm01_encoding_token()
    _series_meta_path(out_series_dir).write_text(json.dumps(meta))

    writer.writerow(_row_for(
        out_path, series_name, idx, spacing, patient_id, meta_src["view"],
    ))
    print(f"ok=true file={file_label} series={series_name} slice={idx} "
          f"view={meta_src['view']} spacing=({spacing[0]:.3f}, {spacing[1]:.3f})")
    return 1


def process_dicom(
    dcm_path: Path,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
    force: bool = False,
) -> int:
    """Convert one DICOM radiograph to a norm16 PNG slice. Returns slices written."""
    try:
        arr, meta_src = load_dicom_image(dcm_path)
    except Exception as e:
        print(f"skip=true file={dcm_path.name} reason=load_failed err={e}")
        return 0
    if not meta_src["patient_id"]:
        print(f"skip=true file={dcm_path.name} reason=no_patient_id")
        return 0
    series_name = f"{dataset_name}_{meta_src['patient_id']}_{meta_src['study_id']}"
    return _process_image(
        arr, meta_src, dcm_path.name, series_name,
        out_root, dataset_name, writer, dcm_path.name, force,
    )


def process_png(
    png_path: Path,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
    force: bool = False,
) -> int:
    """Convert one PNG radiograph to a norm16 PNG slice. Returns slices written."""
    try:
        arr, meta_src = load_png_image(png_path)
    except Exception as e:
        print(f"skip=true file={png_path.name} reason=load_failed err={e}")
        return 0
    if not meta_src["patient_id"]:
        print(f"skip=true file={png_path.name} reason=no_patient_id")
        return 0
    series_name = f"{dataset_name}_{meta_src['patient_id']}_{meta_src['study_id']}"
    return _process_image(
        arr, meta_src, png_path.name, series_name,
        out_root, dataset_name, writer, png_path.name, force,
    )


def find_dicom_files(search_dir: Path) -> list[Path]:
    files = [p for p in search_dir.rglob("*.dcm") if not p.name.startswith("._")]
    files.sort(key=lambda p: (str(p.parent), p.name))
    return files


def find_png_files(search_dir: Path) -> list[Path]:
    files = [p for p in search_dir.rglob("*.png") if not p.name.startswith("._")]
    files.sort(key=lambda p: (str(p.parent), p.name))
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert radiographs (DICOM or PNG) to norm16 16-bit PNG "
                    "slices with index CSV."
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dicom-dir", type=Path, default=None,
                      help="Directory tree containing .dcm radiographs")
    mode.add_argument("--png-dir", type=Path, default=None,
                      help="Directory tree containing radiograph PNGs "
                           "(CheXpert layout Patient/study_view.png)")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"),
                    help="Root output directory for processed data")
    ap.add_argument("--dataset-name", type=str, required=True,
                    help="Dataset identifier (e.g., mimic-cxr, padchest, chexpert)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Process at most N images (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess images even if already ingested")
    args = ap.parse_args(argv)

    input_dir = args.dicom_dir if args.dicom_dir is not None else args.png_dir
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    if args.dicom_dir is not None:
        files = find_dicom_files(input_dir)
        if not files:
            print(f"ERROR: no .dcm files found in: {input_dir}", file=sys.stderr)
            return 2
    else:
        files = find_png_files(input_dir)
        if not files:
            print(f"ERROR: no .png files found in: {input_dir}", file=sys.stderr)
            return 2

    print(f"Found {len(files)} radiograph files in {input_dir}")

    index_path = args.out_root / args.dataset_name / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    total_images = 0

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])

        for fp in files:
            if args.dicom_dir is not None:
                n = process_dicom(fp, args.out_root, args.dataset_name, w,
                                  force=args.force)
            else:
                n = process_png(fp, args.out_root, args.dataset_name, w,
                                force=args.force)
            if n > 0:
                total_slices += n
                total_images += 1
            if args.max_images and total_images >= args.max_images:
                break

    print(f"\nDone: {total_images} images, {total_slices} slices")
    print(f"Index: {index_path}")

    if total_slices == 0:
        print("ERROR: no slices ingested — check patient identity and input format",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

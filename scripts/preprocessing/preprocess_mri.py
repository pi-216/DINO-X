#!/usr/bin/env python3
"""MRI NIfTI volumes -> z-score-normalized 16-bit PNG slices.

Ingests MRI datasets (AMOS MRI subset, BraTS, fastMRI after DICOM->NIfTI
conversion) into the same slice-PNG + index-CSV layout as the CT pipelines,
with MRI-specific handling:

  - MRI intensities are coil-dependent, so each acquired sequence volume is
    z-scored (zero mean, unit variance) and stored with the canonical
    `zscore16_png` encoding from zoo.encoding (never inlined here).
  - Volumes are reoriented to RAS+ before axial slicing, so sagittal/coronal
    acquisitions slice along the superior-inferior axis like axial ones.
  - Series identity: BraTS `_0000.._0003` suffixes map to t1/t1ce/t2/t2flair;
    named tokens (t1, t2, t2w, pdfs, ...) are recognized; `--single-sequence`
    labels datasets with no per-file sequence encoding (e.g. AMOS MRI).
    Volumes with an unrecognized sequence are skipped so segmentation label
    files are never ingested as images.
  - patient_id per row (BraTS scan suffixes fold into the case ID) so split
    builders can enforce zoo.splits leakage asserts.
  - Each series directory carries meta.json (spacing, patient, sequence,
    normalization stats); a rerun without --force re-indexes from it.

Usage:
    python scripts/preprocessing/preprocess_mri.py \
        --nifti-dir data/raw/amos/AMOS22-MRI \
        --out-root data/processed \
        --dataset-name amos-mri \
        --single-sequence mri
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

from zoo.encoding import (
    zscore_encoding_token,
    zscore_to_u16,
)

BRATS_SEQ_BY_CODE = {0: "t1", 1: "t1ce", 2: "t2", 3: "t2flair"}
SEQ_TOKENS = (
    "t1ce", "t2flair", "t1w", "t2w", "t2f", "flair",
    "pdfs", "dwi", "adc", "pd", "pre", "post", "t1", "t2",
)
SEQ_TOKEN_RE = re.compile(
    r"^(.*)_(" + "|".join(SEQ_TOKENS) + r")$", re.IGNORECASE
)
BRATS_SUFFIX_RE = re.compile(r"(?i)^(brats[_-]?\d+(?:[_-]\d+)?)_000([0-3])$")
BRATS_PATIENT_RE = re.compile(r"(?i)^brats[_-]?(\d+)(?:[_-]\d+)?$")
AMOS_PATIENT_RE = re.compile(r"(?i)^amos[_-]?(\d+)$")


def detect_sequence(stem: str) -> tuple[str, str | None]:
    """Split a NIfTI filename stem into (base_stem, sequence | None)."""
    m = BRATS_SUFFIX_RE.match(stem)
    if m:
        return m.group(1), BRATS_SEQ_BY_CODE[int(m.group(2))]
    m = SEQ_TOKEN_RE.match(stem)
    if m:
        return m.group(1), m.group(2).lower()
    return stem, None


def derive_patient_id(base_stem: str) -> str:
    """Map a base (sequence-stripped) stem to a patient/case identity."""
    m = BRATS_PATIENT_RE.match(base_stem)
    if m:
        return f"BraTS_{m.group(1)}"
    m = AMOS_PATIENT_RE.match(base_stem)
    if m:
        return f"amos_{m.group(1)}"
    return base_stem


def load_volume(nifti_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume, reorient to RAS+, return ((S, A, R), spacings)."""
    try:
        import nibabel as nib
    except ImportError:
        raise SystemExit("Missing dependency: nibabel. Install with: pip install nibabel")

    img = nib.as_closest_canonical(nib.load(str(nifti_path)))
    data = np.asarray(img.dataobj, dtype=np.float32)

    pixdim = img.header["pixdim"]
    spacing_x = float(abs(pixdim[1]))
    spacing_y = float(abs(pixdim[2]))
    spacing_z = float(abs(pixdim[3]))
    spacing_x = spacing_x if spacing_x > 0 else 1.0
    spacing_y = spacing_y if spacing_y > 0 else 1.0
    spacing_z = spacing_z if spacing_z > 0 else 1.0

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Unexpected NIfTI dimensions: {data.ndim} in {nifti_path}")

    volume = np.transpose(data, (2, 1, 0))
    return volume, (spacing_x, spacing_y, spacing_z)


def zscore_normalize(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-volume z-score (zero mean, unit variance); constant volumes -> zeros."""
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    if std <= 1e-8:
        return np.zeros_like(volume), mean, std
    return (volume - mean) / std, mean, std


def save_slice_zscore16(z_slice: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(zscore_to_u16(z_slice)).save(out_path, format="PNG")


def _series_meta_path(series_dir: Path) -> Path:
    return series_dir / "meta.json"


def _row_for(
    png_path: Path,
    series_name: str,
    slice_idx: int,
    spacing: tuple[float, float, float],
    patient_id: str,
    sequence: str,
) -> list[str]:
    return [
        str(png_path),
        series_name,
        str(slice_idx),
        zscore_encoding_token(),
        f"{spacing[0]:.6f}",
        f"{spacing[1]:.6f}",
        f"{spacing[2]:.6f}",
        patient_id,
        sequence,
    ]


def _reindex_existing(
    series_dir: Path,
    series_name: str,
    writer: csv.writer,
) -> int:
    meta_file = _series_meta_path(series_dir)
    if not meta_file.exists():
        return 0
    meta = json.loads(meta_file.read_text())
    spacing = tuple(meta["spacing"])
    n = 0
    for p in sorted(series_dir.glob("slice_*.png")):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except ValueError:
            continue
        writer.writerow(_row_for(
            p, series_name, idx, spacing, meta["patient_id"], meta["sequence"],
        ))
        n += 1
    return n


def process_volume(
    nifti_path: Path,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
    single_sequence: str | None = None,
    force: bool = False,
    min_slices: int = 3,
) -> int:
    """Convert one MRI volume to z-score PNG slices. Returns slices written."""
    stem = nifti_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    base, sequence = detect_sequence(stem)
    if sequence is None:
        sequence = single_sequence
    if sequence is None:
        print(f"skip=true file={nifti_path.name} reason=unrecognized_sequence")
        return 0

    patient_id = derive_patient_id(base)
    series_name = f"{dataset_name}_{base}_{sequence}"
    out_series_dir = out_root / dataset_name / series_name

    if not force and out_series_dir.exists():
        n = _reindex_existing(out_series_dir, series_name, writer)
        if n > 0:
            print(f"skip=true file={nifti_path.name} reason=already_converted slices={n}")
            return n

    try:
        volume, spacing = load_volume(nifti_path)
    except Exception as e:
        print(f"skip=true file={nifti_path.name} reason=load_failed err={e}")
        return 0

    if volume.shape[0] < min_slices:
        print(f"skip=true file={nifti_path.name} reason=too_few_slices slices={volume.shape[0]}")
        return 0

    zvol, mean, std = zscore_normalize(volume)
    n_slices = volume.shape[0]
    for i in range(n_slices):
        out_path = out_series_dir / f"slice_{i:04d}.png"
        save_slice_zscore16(zvol[i], out_path)
        writer.writerow(_row_for(
            out_path, series_name, i, spacing, patient_id, sequence,
        ))

    _series_meta_path(out_series_dir).write_text(json.dumps({
        "spacing": list(spacing),
        "patient_id": patient_id,
        "sequence": sequence,
        "mean": mean,
        "std": std,
        "encoding": zscore_encoding_token(),
    }))

    print(f"ok=true file={nifti_path.name} slices={n_slices} sequence={sequence} "
          f"spacing=({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f})")
    return n_slices


def find_nifti_files(search_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.nii.gz", "*.nii"):
        files.extend(p for p in search_dir.rglob(pattern) if not p.name.startswith("._"))
    files.sort(key=lambda p: p.name)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert MRI NIfTI volumes to z-score 16-bit PNG slices with index CSV."
    )
    ap.add_argument("--nifti-dir", type=Path, required=True,
                    help="Directory containing .nii.gz files")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"),
                    help="Root output directory for processed data")
    ap.add_argument("--dataset-name", type=str, required=True,
                    help="Dataset identifier (e.g., amos-mri, brats-2021, fastmri)")
    ap.add_argument("--single-sequence", type=str, default=None,
                    help="Label every volume with this sequence when the filename "
                         "carries no sequence token (e.g. AMOS MRI: 'mri')")
    ap.add_argument("--max-volumes", type=int, default=0,
                    help="Process at most N volumes (0 = all)")
    ap.add_argument("--min-slices", type=int, default=3,
                    help="Skip volumes with fewer axial slices than this")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess volumes even if output already exists")
    args = ap.parse_args(argv)

    nifti_dir = args.nifti_dir
    if not nifti_dir.is_dir():
        print(f"ERROR: NIfTI directory not found: {nifti_dir}", file=sys.stderr)
        return 2

    nifti_files = find_nifti_files(nifti_dir)
    if not nifti_files:
        print(f"ERROR: no .nii.gz or .nii files found in: {nifti_dir}", file=sys.stderr)
        return 2

    print(f"Found {len(nifti_files)} NIfTI files in {nifti_dir}")

    index_path = args.out_root / args.dataset_name / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    total_volumes = 0

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                    "spacing_x", "spacing_y", "spacing_z", "patient_id", "sequence"])

        for nf in nifti_files:
            n = process_volume(
                nf, args.out_root, args.dataset_name, w,
                single_sequence=args.single_sequence,
                force=args.force,
                min_slices=args.min_slices,
            )
            if n > 0:
                total_slices += n
                total_volumes += 1
            if args.max_volumes and total_volumes >= args.max_volumes:
                break

    print(f"\nDone: {total_volumes} volumes, {total_slices} slices")
    print(f"Index: {index_path}")

    if total_slices == 0:
        print("ERROR: no slices ingested — check --single-sequence or sequence tokens",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

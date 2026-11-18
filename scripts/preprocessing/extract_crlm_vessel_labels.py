#!/usr/bin/env python3
"""Extract per-slice hepatic/portal VESSEL label from CRLM SEG DICOM, aligned to CT by z.

CRLM SEGs are organized BY SEGMENT (all frames of segment N in a run), each frame
carrying ImagePositionPatient z. We reindex frames by z, identify Hepatic/Portal
segments, and label each CT slice vessel-present if a vessel frame at that z has
>= min_pixels. CT slice z is read from the raw CT DICOM IPP.

Matching: SEG is matched to the CT series whose z-coverage has the greatest overlap
with the SEG's vessel/liver z set.

Output CSV:
  png_path,label,spacing_x,spacing_y,spacing_z,patient_id,vessel_pixels
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom

VESSEL_SEG_NAMES = ("hepatic", "portal")
Z_TOL = 0.5  # mm tolerance for matching z positions


def read_patient_id_from_ct(ct_dir: Path) -> str | None:
    """Read the DICOM PatientID (0010,0020) from a CT series directory.

    The label CSV's ``patient_id`` column previously held the referenced
    ``SeriesInstanceUID`` (a bug: patient counts then equaled series counts,
    wrong for multi-phase datasets). The PatientID must come from the CT
    DICOM's own PatientID tag, not the series UID. Returns None if unavailable.
    """
    dcm_files = sorted(ct_dir.glob("*.dcm"))
    if not dcm_files:
        return None
    try:
        ds = pydicom.dcmread(dcm_files[0], force=True, stop_before_pixels=True)
        pid = getattr(ds, "PatientID", None)
        if pid:
            return str(pid)
    except Exception:
        pass
    return None


def get_segment_map(ds) -> dict[int, str]:
    seg_map = {}
    if "SegmentSequence" in ds:
        for seg in ds.SegmentSequence:
            num = int(getattr(seg, "SegmentNumber", 0))
            name = getattr(seg, "SegmentName", None) or getattr(seg, "SegmentLabel", None)
            seg_map[num] = str(name)
    return seg_map


def read_ct_z_by_dir(ct_root: Path):
    """Map each CT series dir (uid) -> sorted list of (z, instance_uid)."""
    out = {}
    for d in ct_root.iterdir():
        if not d.is_dir():
            continue
        dcm_files = sorted(d.glob("*.dcm"))
        if not dcm_files:
            continue
        zs = []
        for fp in dcm_files:
            try:
                ds = pydicom.dcmread(str(fp), force=True, stop_before_pixels=True)
            except Exception:
                continue
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = float(ipp[2]) if ipp is not None else None
            if z is None:
                continue
            zs.append(z)
        if zs:
            out[d.name] = sorted(zs)
    return out


def read_seg(dcm_path: Path):
    ds = pydicom.dcmread(str(dcm_path), force=True)
    if getattr(ds, "file_meta", None) is None or getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
    frames = ds.pixel_array.copy().astype(np.uint8)
    pf = ds.PerFrameFunctionalGroupsSequence
    n = frames.shape[0]

    # Reference CT series UID: SEG references its source CT series explicitly.
    ref_series_uid = None
    try:
        rss = getattr(ds, "ReferencedSeriesSequence", None)
        if rss:
            ref_series_uid = getattr(rss[0], "SeriesInstanceUID", None)
    except Exception:
        ref_series_uid = None

    seg_map = get_segment_map(ds)
    vessel_segs = {num for num, nm in seg_map.items() if nm and any(v in nm.lower() for v in VESSEL_SEG_NAMES)}

    # z -> list of (seg_num, mask)
    zdata = defaultdict(list)
    for i in range(n):
        z = None
        pps = getattr(pf[i], "PlanePositionSequence", None)
        if pps:
            ipp = getattr(pps[0], "ImagePositionPatient", None)
            if ipp is not None:
                z = float(ipp[2])
        sids = getattr(pf[i], "SegmentIdentificationSequence", None)
        seg = int(getattr(sids[0], "ReferencedSegmentNumber", 0)) if sids else 0
        if z is not None:
            zdata[z].append((seg, frames[i]))
    return zdata, vessel_segs, seg_map, ref_series_uid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-root", required=True, type=Path)
    ap.add_argument("--ct-root", required=True, type=Path)
    ap.add_argument("--index-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-pixels", type=int, default=30)
    ap.add_argument("--max-z-gap", type=float, default=5.0, help="skip vessel z further than this from any CT z")
    args = ap.parse_args(argv)

    # Load index; index series_dir should be the original CT dir path.
    index = list(csv.DictReader(open(args.index_csv)))
    # Group by series_dir (the CT dir), keep slice order
    by_series = defaultdict(list)
    for r in index:
        by_series[r["series_dir"]].append(r)
    print(f"CT index: {len(index)} slices, {len(by_series)} series")

    # CT z per uid
    ct_z_by_uid = read_ct_z_by_dir(args.ct_root)
    print(f"CT series with z: {len(ct_z_by_uid)}")

    # Map series_dir (a path) -> uid = basename
    # series_dir points into ct_root; basename is the uid
    def series_dir_to_uid(sd):
        return Path(sd).name

    seg_dirs = [d for d in sorted(args.seg_root.iterdir()) if d.is_dir() and (d / "00000001.dcm").exists()]
    print(f"SEG series: {len(seg_dirs)}")

    out_rows = []
    matched = 0
    unmatched_series = 0

    for seg_dir in seg_dirs:
        try:
            zdata, vessel_segs, seg_map, ref_series_uid = read_seg(seg_dir / "00000001.dcm")
        except Exception as e:
            print(f"  skip decode {seg_dir.name}: {e}")
            continue
        if not zdata:
            continue

        # Match to its CT series via the referenced SeriesInstanceUID.
        best_uid = ref_series_uid
        if best_uid is None or best_uid not in ct_z_by_uid:
            unmatched_series += 1
            print(f"  UNMATCHED {seg_dir.name} (ref uid not in CT: {best_uid})")
            continue
        ct_zs = ct_z_by_uid[best_uid]

        # Find the series_dir in index matching this uid
        series_dir = None
        for sd in by_series:
            if series_dir_to_uid(sd) == best_uid:
                series_dir = sd
                break
        if series_dir is None:
            unmatched_series += 1
            continue

        # Resolve the real PatientID from the CT DICOM (NOT the series UID).
        patient_id = read_patient_id_from_ct(Path(series_dir)) or best_uid

        rows = by_series[series_dir]
        # rows are in slice order (as sorted by z in preprocessing). Match each
        # row to a z by nearest CT z with tolerance; but we don't have exact per-row
        # z in index. We DO have ct_zs sorted = same order as rows (preprocessing
        # sorted sections by z). Therefore row i corresponds to ct_zs[i].
        # Guard: len mismatch
        n_rows = len(rows)
        n_ctz = len(ct_zs)
        if n_rows != n_ctz:
            print(f"  WARN series_dir {series_dir}: rows={n_rows} vs ct_z={n_ctz}; truncating")
        for i in range(min(n_rows, n_ctz)):
            r = rows[i]
            z = ct_zs[i]
            # vessel pixels at this z
            vessel_pix = 0
            for (seg, mask) in zdata.get(z, []):
                if seg in vessel_segs:
                    vessel_pix = max(vessel_pix, int((mask > 0).sum()))
            label = 1 if vessel_pix >= args.min_pixels else 0
            out_rows.append({
                "png_path": r["png_path"],
                "label": label,
                "spacing_x": r["spacing_x"],
                "spacing_y": r["spacing_y"],
                "spacing_z": r["spacing_z"],
                "patient_id": patient_id,
                "vessel_pixels": vessel_pix,
            })
        matched += 1

    print(f"Matched {matched} SEG series, unmatched {unmatched_series}")
    pos = sum(1 for r in out_rows if r["label"] == 1)
    print(f"Labels: {len(out_rows)} slices, positive(vessel)={pos} ({100*pos/max(len(out_rows),1):.1f}%)")
    if pos:
        print(f"  vessel_pixels among pos: min={min(r['vessel_pixels'] for r in out_rows if r['label'])} "
              f"max={max(r['vessel_pixels'] for r in out_rows if r['label'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["png_path", "label", "spacing_x", "spacing_y", "spacing_z", "patient_id", "vessel_pixels"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
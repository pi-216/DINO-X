#!/usr/bin/env python3
"""Phase 4: Create a deterministic train/validation split manifest.

This operates on Phase 2's `data/processed/_index/index.csv` and produces a small
JSON manifest intended to be referenced by training/monitoring entrypoints.

Design notes:
- The index groups slices by `series_dir`. For MSD-style datasets one series ==
  one patient, so a series-level split is patient-safe *only when that holds*.
  It does NOT hold for multi-series-per-patient corpora (LIDC-IDRI, CRLM
  multi-phase): a patient with several series can land in BOTH train and val,
  leaking patient identity. Before trusting the split for such a dataset, map
  series -> patient (see ``zoo.splits``) and split at patient granularity, then
  run ``zoo.splits.assert_no_series_leak``.
- The manifest stores split membership as `series_dir` strings (not per-slice
  lists) to keep it compact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# repo root on path so zoo.splits is importable when run standalone
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from zoo.splits import assert_no_series_leak  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument(
        "--patient-map", type=Path, default=None,
        help="Optional CSV mapping series_dir -> patient_id (columns: "
             "'series_dir','patient_id'). When provided, the split is made at "
             "PATIENT granularity (a patient's all series stay on one side) and "
             "series-level leakage is asserted. "
             "ALTERNATIVELY, an index with a 'patient_id' column is treated the "
             "same way with no map file needed.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/processed/_splits/val10_seed<seed>.json",
    )
    args = ap.parse_args()

    if not args.index_csv.exists():
        raise SystemExit(f"index_csv not found: {args.index_csv}")

    if not (0.0 < args.val_frac < 1.0):
        raise SystemExit("--val-frac must be in (0,1)")

    import random

    # ---- resolve patient identity -------------------------------------------------
    # patient_id per series_dir. Sources (priority): --patient-map file, then a
    # 'patient_id' column present in the index itself. If neither is available,
    # we fall back to series-level grouping WITH a warning that patient-safe
    # splitting is unverified (only valid for one-series-per-patient datasets).
    patient_of_series: dict[str, str] = {}
    has_patient_id_col = False

    if args.patient_map is not None:
        if not args.patient_map.exists():
            raise SystemExit(f"patient_map not found: {args.patient_map}")
        with args.patient_map.open(newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None or not {"series_dir", "patient_id"} <= set(r.fieldnames):
                raise SystemExit(
                    f"patient_map must have 'series_dir' and 'patient_id' columns; "
                    f"found {r.fieldnames}"
                )
            for row in r:
                patient_of_series[row["series_dir"]] = row["patient_id"]

    series_counts: dict[str, int] = {}
    total_rows = 0
    index_patient_of_series: dict[str, str] = {}

    with args.index_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if "series_dir" not in fields:
            raise SystemExit(f"index_csv missing series_dir column: {args.index_csv}")
        has_patient_id_col = "patient_id" in fields
        for row in r:
            total_rows += 1
            sd = row.get("series_dir", "")
            series_counts[sd] = series_counts.get(sd, 0) + 1
            if has_patient_id_col:
                pid = row.get("patient_id", "").strip()
                if pid:
                    index_patient_of_series[sd] = pid

    if not series_counts:
        raise SystemExit(f"No rows/series found in index_csv: {args.index_csv}")

    # ---- decide group key ---------------------------------------------------------
    patient_series: dict[str, list[str]] | None = None
    if patient_of_series:
        # --patient-map wins; any series in the index not covered by the map is
        # treated as its own singleton patient (never silently pooled).
        group_key = "patient_id"
        patient_series = {}
        for sd in series_counts:
            pid = patient_of_series.get(sd)
            if pid is None:
                pid = f"__series__{sd}"  # unmapped series -> unique singleton
            patient_series.setdefault(pid, []).append(sd)
    elif has_patient_id_col:
        group_key = "patient_id"
        patient_series = {}
        for sd in series_counts:
            pid = index_patient_of_series.get(sd)
            if pid is None:
                pid = f"__series__{sd}"
            patient_series.setdefault(pid, []).append(sd)
    else:
        group_key = "series_dir"
        print(
            "WARNING: no patient_id column and no --patient-map; splitting at "
            "series_dir granularity. Patient-safe ONLY if one series == one "
            "patient (MSD). LIDC/CRLM-style multi-series corpora will LEAK "
            "patient identity. Provide --patient-map for a patient-safe split.",
        )

    # ---- shuffle + split groups ---------------------------------------------------
    if patient_series is not None:
        groups = list(patient_series.keys())
    else:
        groups = list(series_counts.keys())

    rng = random.Random(args.seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_val = max(1, int(math.ceil(args.val_frac * n_groups)))
    val_groups = sorted(groups[:n_val])
    train_groups = sorted(groups[n_val:])

    # ---- series-level leakage assertion (patient-aware path only) ----------------
    if group_key == "patient_id":
        train_series = set()
        val_series = set()
        for g in train_groups:
            train_series.update(patient_series[g])
        for g in val_groups:
            val_series.update(patient_series[g])
        assert_no_series_leak(train_series, val_series)

    # ---- counts --------------------------------------------------------------------
    if group_key == "patient_id":
        train_series_all = [s for g in train_groups for s in patient_series[g]]
        val_series_all = [s for g in val_groups for s in patient_series[g]]
        train_images = sum(series_counts[s] for s in train_series_all)
        val_images = sum(series_counts[s] for s in val_series_all)
    else:
        train_images = sum(series_counts[g] for g in train_groups)
        val_images = sum(series_counts[g] for g in val_groups)

    out = args.out
    if out is None:
        out = Path("data/processed/_splits") / f"val10_seed{args.seed}.json"

    membership_key = "series_dir"
    if group_key == "patient_id":
        train_membership = sorted({s for g in train_groups for s in patient_series[g]})
        val_membership = sorted({s for g in val_groups for s in patient_series[g]})
    else:
        train_membership = train_groups
        val_membership = val_groups

    payload = {
        "kind": "dataset_split",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_csv": str(args.index_csv),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "group_key": group_key,
        "counts": {
            "groups_total": n_groups,
            "groups_train": len(train_groups),
            "groups_val": len(val_groups),
            "images_total": total_rows,
            "images_train": train_images,
            "images_val": val_images,
        },
        "train": {membership_key: train_membership},
        "val": {membership_key: val_membership},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("ok=true")
    print(f"out={out}")
    print(f"group_key={group_key}")
    print(f"groups_total={n_groups}")
    print(f"images_total={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

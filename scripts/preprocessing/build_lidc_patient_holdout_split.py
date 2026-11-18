#!/usr/bin/env python3
"""Build a patient-held-out LIDC pretraining split (Release #2).

WHY THIS EXISTS
---------------
The Phase 6 LIDC specialist (LoRA AUROC 0.728, retracted 2026-09-20) was trained on
the *entire* LIDC corpus with a **series-level** split, so the backbone saw the very
patients later used for LoRA evaluation (transductive leakage). This script builds the
corrected split:

  * the LoRA-eval patients (malignancy ``val`` + ``test``) are held out of pretraining
    ENTIRELY — their series never appear in the pretraining index;
  * the remaining series are split train/val **by patient** (``zoo.splits.patient_split``),
    so no patient (and therefore no series) crosses the boundary;
  * every claim is asserted, not assumed — the script fails closed.

Outputs (schema matches what ``scripts/phase5_big_run.py`` consumes — it excludes
``val.series_dir`` from training and treats the rest of the index as train):

  data/mvp/lidc_only_t2_patientholdout.csv   filtered slice index (holdout excluded)
  data/mvp/split_manifest_lidc_patientholdout.json

Runs with the standard library only (``zoo.splits.read_patient_id_file`` has a
dependency-free DICOM reader), so it works on any box with the repo + NAS mounted.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zoo.splits import (  # noqa: E402
    assert_no_patient_leak,
    assert_no_series_leak,
    patient_split,
    read_patient_id_file,
)


def series_to_patient(series_dir: str, raw_root: str) -> str | None:
    """Map a pretraining ``series_dir`` ('data_raw_<UID>') to its DICOM PatientID."""
    uid = series_dir.replace("data_raw_", "", 1)
    d = os.path.join(raw_root, uid)
    if not os.path.isdir(d):
        return None
    dcms = sorted(glob.glob(os.path.join(d, "*.dcm")))
    if not dcms:
        return None
    return read_patient_id_file(dcms[0])


def load_patients(labels_dir: str, split: str) -> set[str]:
    fp = os.path.join(labels_dir, f"malignancy_{split}.csv")
    with open(fp, newline="") as f:
        return {r["patient_id"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/tim/source/activity/DINO-X")
    ap.add_argument("--raw-root", default="/mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw")
    ap.add_argument("--labels-dir", default="/mnt/nas-ai-models/training-data/dino-x/lidc-idri/labels")
    ap.add_argument("--index-csv", default=None)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    idx = a.index_csv or os.path.join(a.repo, "data/mvp/lidc_only_t2.csv")
    out_csv = a.out or os.path.join(a.repo, "data/mvp/lidc_only_t2_patientholdout.csv")
    out_man = os.path.join(a.repo, "data/mvp/split_manifest_lidc_patientholdout.json")

    holdout = load_patients(a.labels_dir, "val") | load_patients(a.labels_dir, "test")

    with open(idx, newline="") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames
        rows = list(rdr)
    assert fields and "series_dir" in fields, f"unexpected index schema: {fields}"

    series = sorted({r["series_dir"] for r in rows})
    sp: dict[str, str | None] = {sd: series_to_patient(sd, a.raw_root) for sd in series}
    unknown = [sd for sd, p in sp.items() if not p]
    holdout_series = {sd for sd, p in sp.items() if p in holdout}

    # Fail closed: a series we cannot attribute to a patient might belong to a
    # holdout patient, so it is excluded from pretraining rather than risk a leak.
    eligible = [sd for sd in series if sp[sd] and sp[sd] not in holdout]

    train_idx, val_idx = patient_split(
        [sp[sd] for sd in eligible], test_size=a.val_frac, seed=a.seed
    )
    train_series = {eligible[i] for i in train_idx}
    val_series = {eligible[i] for i in val_idx}

    # ---- assertions (fail closed) ----
    assert_no_series_leak(train_series, val_series)
    assert_no_patient_leak({sp[s] for s in train_series}, {sp[s] for s in val_series})
    leak = holdout & ({sp[s] for s in train_series} | {sp[s] for s in val_series})
    assert not leak, f"HOLDOUT PATIENTS IN PRETRAIN SPLIT: {sorted(leak)[:5]}"
    assert not (holdout_series & (train_series | val_series)), "holdout series leaked"

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r["series_dir"] in train_series or r["series_dir"] in val_series:
                w.writerow(r)

    manifest = {
        "kind": "split_manifest",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_csv": os.path.relpath(out_csv, a.repo),
        "seed": a.seed,
        "val_frac": a.val_frac,
        "group_key": "patient_id",
        "design": (
            "patient-held-out: the LoRA-eval patients (malignancy val+test) are excluded "
            "from pretraining entirely; remaining series split train/val by patient. "
            "Replaces the series-level split that leaked eval patients (retraction 2026-09-20)."
        ),
        "counts": {
            "train_series": len(train_series),
            "val_series": len(val_series),
            "holdout_series": len(holdout_series),
            "holdout_patients": len(holdout),
            "unattributed_series_excluded": len(unknown),
        },
        "train": {"series_dir": sorted(train_series)},
        "val": {"series_dir": sorted(val_series)},
        "holdout": {
            "series_dir": sorted(holdout_series),
            "patients": sorted(holdout),
            "reason": "malignancy val+test patients — backbone must never see these",
        },
    }
    with open(out_man, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"series total        : {len(series)}")
    print(f"patients (all)      : {len({p for p in sp.values() if p})}")
    print(f"holdout patients    : {len(holdout)}  (malignancy val+test)")
    print(f"holdout series      : {len(holdout_series)}  ({100*len(holdout_series)/len(series):.1f}% of corpus)")
    print(f"unattributed series : {len(unknown)} (excluded, fail-closed)")
    print(f"train series        : {len(train_series)}  ({len({sp[s] for s in train_series})} patients)")
    print(f"val series          : {len(val_series)}  ({len({sp[s] for s in val_series})} patients)")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

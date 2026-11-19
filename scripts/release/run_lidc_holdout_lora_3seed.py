#!/usr/bin/env python3
"""3-seed patient-holdout LIDC malignancy LoRA benchmark (PRD-0001 T1).

Runs the standard LIDC malignancy LoRA protocol (``scripts/finetune_lora.py``)
across seeds 42/43/44 for the pre-registered checkpoints of the patient-held-out
pretraining run (20K pre-dip peak, 40K post-recovery peak, 50K terminal
reference), evaluates each adapter on the never-seen patient test split, and
aggregates the per-seed test AUROCs with 95% CI into the results JSON that T2
adjudicates.

Fail-closed leak verification runs before any training: pretraining patients
(derived from the split manifest's series via DICOM PatientID) must not overlap
the malignancy val/test patients, the manifest holdout list must match the label
CSVs exactly, and the two series sets must be disjoint.

Launch on the 4090 through the scheduler (request the job first, then):

    ./bin/gpu_run --job-id lidc-holdout-lora-3seed-v1 -- \\
        .venv/bin/python3 scripts/release/run_lidc_holdout_lora_3seed.py \\
        --repo /home/tim/source/activity/DINO-X

Exit 0 only when every (checkpoint, seed) cell completed and the results JSON
was written; any failed cell exits 1 after writing the partial JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUN_SLUG = "20261001_182138_lidc-specialist-vit-base-patient-holdout"
PRIMARY_STEPS = (20000, 40000)
REFERENCE_STEPS = (50000,)
SEEDS = (42, 43, 44)
GO_MIN_MEAN = 0.720
BASELINE = 0.710
NAS_LABELS = "/mnt/nas-ai-models/training-data/dino-x/lidc-idri/labels"
NAS_RAW = "/mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw"
OUT_JSON = "results/lidc_patient_holdout_lora_3seed.json"


def aggregate_auroc(values: list[float]) -> dict:
    if not values:
        raise ValueError("aggregate_auroc requires at least one value")
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    se = std / (n ** 0.5)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def classify_gate(mean: float, ci_low: float) -> str:
    if mean >= GO_MIN_MEAN and ci_low > BASELINE:
        return "GO"
    if mean < BASELINE:
        return "FAIL"
    return "PIVOT"


def read_label_patients(labels_dir: Path, split: str) -> set[str]:
    csv_path = labels_dir / f"malignancy_{split}.csv"
    with csv_path.open(newline="") as f:
        patients = {row["patient_id"].strip() for row in csv.DictReader(f)}
    patients.discard("")
    return patients


def verify_no_patient_leak(manifest_path: Path, labels_dir: Path, raw_root: Path) -> dict:
    from zoo.splits import assert_no_patient_leak, read_patient_id_file

    manifest = json.loads(manifest_path.read_text())
    train_series = set(manifest["train"]["series_dir"])
    val_series = set(manifest["val"]["series_dir"])
    holdout_series = set(manifest["holdout"]["series_dir"])
    manifest_holdout_patients = set(manifest["holdout"]["patients"])

    overlap_series = (train_series | val_series) & holdout_series
    if overlap_series:
        raise AssertionError(
            f"holdout series leaked into pretrain split: {len(overlap_series)} "
            f"(e.g. {sorted(overlap_series)[:3]})"
        )

    label_patients = read_label_patients(labels_dir, "val") | read_label_patients(labels_dir, "test")
    if manifest_holdout_patients != label_patients:
        only_manifest = sorted(manifest_holdout_patients - label_patients)[:5]
        only_labels = sorted(label_patients - manifest_holdout_patients)[:5]
        raise ValueError(
            "manifest holdout patients and malignancy val+test CSV patients disagree: "
            f"manifest-only={only_manifest} label-only={only_labels}"
        )

    pretrain_patients: set[str] = set()
    for series_dir in sorted(train_series | val_series):
        uid = series_dir.removeprefix("data_raw_")
        series_path = raw_root / uid
        dcms = sorted(series_path.glob("*.dcm")) if series_path.is_dir() else []
        patient = read_patient_id_file(dcms[0]) if dcms else None
        if not patient:
            raise RuntimeError(
                f"cannot attribute pretrain series to a patient, refusing to assume: {series_dir}"
            )
        pretrain_patients.add(patient)

    assert_no_patient_leak(pretrain_patients, label_patients)
    overlap = len(pretrain_patients & label_patients)
    return {
        "pretrain_patients": len(pretrain_patients),
        "holdout_patients": len(label_patients),
        "pretrain_series": len(train_series | val_series),
        "holdout_series": len(holdout_series),
        "overlap": overlap,
        "manifest": str(manifest_path),
    }


def build_finetune_cmd(
    python: str,
    finetune_script: Path,
    checkpoint: Path,
    train_csv: Path,
    val_csv: Path,
    data_root: Path,
    output: Path,
    seed: int,
) -> list[str]:
    return [
        python, str(finetune_script),
        "--backbone", str(checkpoint),
        "--train-csv", str(train_csv),
        "--val-csv", str(val_csv),
        "--data-root", str(data_root),
        "--task", "classification",
        "--num-classes", "2",
        "--rank", "8",
        "--alpha", "16",
        "--lr", "5e-4",
        "--epochs", "50",
        "--batch-size", "32",
        "--patience", "10",
        "--es-metric", "auroc",
        "--window-level", "-30",
        "--window-width", "120",
        "--device", "cuda",
        "--seed", str(seed),
        "--output", str(output),
    ]


def checkpoint_slug(step: int) -> str:
    return f"checkpoint_{step:08d}.pth"


def adapter_slug(step: int, seed: int) -> str:
    return f"lidc-holdout-lora-ck{step}-seed{seed}"


def evaluate_adapter(repo: Path, checkpoint: Path, adapter_dir: Path, test_csv: Path) -> dict:
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(repo / "scripts"))
    from finetune_lora import LabeledImageDataset, _load_csv

    from zoo.hub import load_model
    from zoo.peft import load_adapter

    cfg = json.loads((adapter_dir / "finetune_config.json").read_text())
    device = torch.device("cuda")
    backbone = load_model(str(checkpoint), device=str(device))
    dim = backbone.dim
    img_size = backbone.img_size
    backbone = load_adapter(backbone, adapter_dir)
    backbone = backbone.to(device)
    head = nn.Linear(dim, cfg["num_classes"]).to(device)
    head.load_state_dict(torch.load(adapter_dir / "head.pth", map_location=device))
    backbone.eval()
    head.eval()

    rows = _load_csv(test_csv)
    ds = LabeledImageDataset(
        rows,
        img_size=img_size,
        input_format="hu16_png",
        window_level=-30.0,
        window_width=120.0,
        augment=False,
        data_root=repo,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    probs: list = []
    labels: list = []
    with torch.no_grad():
        for images, spacings, targets in loader:
            images = images.to(device)
            spacing = spacings.to(device) if cfg["scale_aware"] else None
            logits = head(backbone(images, spacing=spacing)[:, 0])
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
            labels.extend(targets.cpu().tolist())

    import numpy as np

    probs_np = np.asarray(probs)
    labels_np = np.asarray(labels)
    from eval_external import auroc, patient_auroc

    pids = np.asarray([row.patient_id for row in rows])
    slice_auc = auroc(probs_np, labels_np)
    patient_auc, n_patients = patient_auroc(probs_np, labels_np, pids)
    return {
        "test_slice_auroc": slice_auc,
        "test_patient_auroc": patient_auc,
        "test_patient_count": n_patients,
        "test_slice_count": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--labels-dir", default=NAS_LABELS)
    ap.add_argument("--raw-root", default=NAS_RAW)
    ap.add_argument("--split-manifest", default="data/mvp/split_manifest_lidc_patientholdout.json")
    ap.add_argument("--run-slug", default=RUN_SLUG)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--steps", default="20000,40000,50000")
    ap.add_argument("--adapters-subdir", default="adapters")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    labels_dir = Path(a.labels_dir)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    steps = [int(s) for s in a.steps.split(",") if s.strip()]

    leak = verify_no_patient_leak(repo / a.split_manifest, labels_dir, Path(a.raw_root))
    print(
        f"leak verification OK: {leak['pretrain_patients']} pretrain / "
        f"{leak['holdout_patients']} holdout patients, overlap {leak['overlap']}",
        flush=True,
    )

    finetune_script = Path(__file__).resolve().parents[1] / "finetune_lora.py"
    rows = []
    for step in steps:
        ckpt = repo / "runs" / a.run_slug / checkpoint_slug(step)
        if not ckpt.exists():
            rows.append({"step": step, "error": f"missing checkpoint {ckpt}"})
            continue
        for seed in seeds:
            out_dir = repo / a.adapters_subdir / adapter_slug(step, seed)
            cmd = build_finetune_cmd(
                python=a.python,
                finetune_script=finetune_script,
                checkpoint=ckpt,
                train_csv=labels_dir / "malignancy_train.csv",
                val_csv=labels_dir / "malignancy_val.csv",
                data_root=repo,
                output=out_dir,
                seed=seed,
            )
            print(f"\n=== step={step} seed={seed} -> {out_dir.name}", flush=True)
            started = time.time()
            proc = subprocess.run(cmd, cwd=repo)
            if proc.returncode != 0:
                rows.append({"step": step, "seed": seed, "error": f"finetune exit {proc.returncode}"})
                continue
            cfg = json.loads((out_dir / "finetune_config.json").read_text())
            row = {
                "step": step,
                "seed": seed,
                "adapter": str(out_dir.relative_to(repo)),
                "best_epoch": cfg["best_epoch"],
                "best_val_auroc": cfg["best_val_metrics"].get("auroc"),
                "seconds": round(time.time() - started, 1),
            }
            row.update(evaluate_adapter(repo, ckpt, out_dir, labels_dir / "malignancy_test.csv"))
            rows.append(row)
            print(
                f"  val_auroc={row['best_val_auroc']:.4f} "
                f"test_slice_auroc={row['test_slice_auroc']:.4f} "
                f"test_patient_auroc={row['test_patient_auroc']:.4f}",
                flush=True,
            )

    aggregates = {}
    for step in steps:
        vals = [r["test_slice_auroc"] for r in rows if r.get("step") == step and "test_slice_auroc" in r]
        if not vals:
            aggregates[str(step)] = {"error": "no completed seeds"}
            continue
        agg = aggregate_auroc(vals)
        agg["gate"] = classify_gate(agg["mean"], agg["ci95_low"])
        agg["primary"] = step in PRIMARY_STEPS
        aggregates[str(step)] = agg

    failed = any("error" in r for r in rows)
    summary = {
        "kind": "lidc_patient_holdout_lora_3seed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_prd": 34,
        "ticket": 35,
        "run_slug": a.run_slug,
        "protocol": {
            "task": "classification", "num_classes": 2, "rank": 8, "alpha": 16,
            "lr": "5e-4", "epochs": 50, "batch_size": 32, "patience": 10,
            "es_metric": "auroc", "window_level": -30, "window_width": 120,
            "input_format": "hu16_png", "scale_aware": "backbone-derived",
            "train_csv": str(labels_dir / "malignancy_train.csv"),
            "val_csv": str(labels_dir / "malignancy_val.csv"),
            "test_csv": str(labels_dir / "malignancy_test.csv"),
        },
        "seeds": seeds,
        "steps": steps,
        "primary_steps": list(PRIMARY_STEPS),
        "reference_steps": list(REFERENCE_STEPS),
        "leak_verification": leak,
        "gate": {
            "go_min_mean": GO_MIN_MEAN,
            "baseline_ci_floor": BASELINE,
            "note": "GO: mean >= 0.720 AND ci95_low > 0.710; FAIL: mean < 0.710; else PIVOT. "
                    "Adjudication + adversarial pass belongs to T2.",
        },
        "rows": rows,
        "aggregates": aggregates,
        "complete": not failed,
    }
    out_path = repo / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 66)
    print(f"{'step':>8} {'n':>3} {'mean':>8} {'std':>8} {'ci_low':>8}  gate")
    for step in steps:
        agg = aggregates[str(step)]
        if "error" in agg:
            print(f"{step:>8}  ERROR")
        else:
            print(
                f"{step:>8} {agg['n']:>3} {agg['mean']:>8.4f} {agg['std']:>8.4f} "
                f"{agg['ci95_low']:>8.4f}  {agg['gate']}"
            )
    print("=" * 66)
    print(f"wrote {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""View-retrieval quality sweep for Release #2 (LIDC patient-holdout pretrain).

Runs ``phase5_view_retrieval_eval.py`` across every checkpoint of the patient-holdout
run AND a control: the (contaminated) Phase 6 50K checkpoint evaluated on the **same**
split with the **same** harness and ``n``. Without the control, any comparison against
the historically reported numbers would be across different eval sets — a different
measurement, not a re-measurement.

Quality assessment only. View retrieval is NOT a release gate for single-organ models
(Phase 8); it is used here to (a) put a number on representation structure, and
(b) select a checkpoint honestly rather than defaulting to 50K after a collapse episode.

Usage:
    python scripts/release/eval_view_retrieval_sweep.py --n 512 --layer 9
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOLDOUT_RUN = "20261001_182138_lidc-specialist-vit-base-patient-holdout"
PHASE6_RUN = "20260723_194257_lidc-specialist-vit-base-scale-aware"


def checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted(run_dir.glob("checkpoint_0*.pth")):
        step = p.stem.split("_")[-1]
        if p.stem.startswith("checkpoint_final"):
            continue
        out.append((f"holdout@{int(step)//1000}k", p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--index-csv", default="data/mvp/lidc_only_t2_patientholdout.csv")
    ap.add_argument("--split-manifest", default="data/mvp/split_manifest_lidc_patientholdout.json")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="results/quality_view_retrieval_lidc_holdout.json")
    a = ap.parse_args()

    runs = Path(a.repo) / "runs"
    targets = checkpoints(runs / HOLDOUT_RUN)
    # control: Phase 6 50K on the SAME split (weights load fine regardless of device)
    ctrl = runs / PHASE6_RUN / "checkpoint_00050000.pth"
    if ctrl.exists():
        targets = targets + [("control:phase6-50k", ctrl)]
    else:
        print(f"WARN: control checkpoint missing: {ctrl}", file=sys.stderr)

    evals = Path(a.repo) / "scripts/phase5_view_retrieval_eval.py"
    rows = []
    for label, ckpt in targets:
        cmd = [
            a.python, str(evals),
            "--checkpoint", str(ckpt),
            "--index-csv", a.index_csv,
            "--split-manifest", a.split_manifest,
            "--scale-aware", "--n", str(a.n), "--vit-layer", str(a.layer),
        ]
        print(f"\n=== {label} :: {ckpt.name}", flush=True)
        r = subprocess.run(cmd, cwd=a.repo, capture_output=True, text=True)
        tail = (r.stdout or "")[-400:]
        print(tail, flush=True)
        if r.returncode != 0:
            print(f"FAILED {label}: {(r.stderr or '')[-400:]}", file=sys.stderr, flush=True)
            rows.append({"label": label, "checkpoint": ckpt.name, "error": "nonzero exit"})
            continue
        layer_tag = f"_layer{a.layer}" if a.layer is not None else ""
        out_json = ckpt.parent / f"view_retrieval_step{int(ckpt.stem.split('_')[-1])}_N{a.n}{layer_tag}.json"
        if not out_json.exists():
            out_json = ckpt.parent / f"view_retrieval_step{ckpt.stem.split('_')[-1]}_N{a.n}.json"
        d = json.loads(out_json.read_text()) if out_json.exists() else {}
        rows.append({
            "label": label,
            "checkpoint": ckpt.name,
            "top1": d.get("top1"),
            "topk_acc": d.get("topk_acc"),
            "ratio_vs_random": d.get("ratio_vs_random"),
            "seconds": round(d.get("seconds", 0), 1),
            "json": str(out_json.relative_to(a.repo)) if out_json.exists() else None,
        })

    summary = {
        "kind": "quality_view_retrieval_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": a.n, "vit_layer": a.layer,
        "index_csv": a.index_csv, "split_manifest": a.split_manifest,
        "note": ("Control = Phase 6 (contaminated) 50K evaluated on the SAME split/harness "
                 "so the comparison is a measurement, not an assumption. View retrieval is "
                 "not a release gate for single-organ models (Phase 8)."),
        "rows": rows,
    }
    out = Path(a.repo) / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 62)
    print(f"{'label':24} {'top1':>8} {'top5':>8} {'ratio':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['label']:24}  ERROR")
        else:
            print(f"{r['label']:24} {r['top1']:>8.4f} {r['topk_acc']:>8.4f} {r['ratio_vs_random']:>7}")
    print("=" * 62)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

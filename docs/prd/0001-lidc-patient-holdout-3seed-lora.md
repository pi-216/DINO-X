---
prd: "0001"
title: "Release #2 — LIDC Patient-Holdout 3-Seed LoRA Evaluation & Verdict"
status: ruled
ruled: "2026-11-18"
intent: "moves:released-models"
---

# PRD-0001 — Release #2: LIDC Patient-Holdout 3-Seed LoRA Evaluation & Verdict

## 1. Problem & Context

1. **Why Phase 6 (`0.728`) was retracted (`2026-09-20`):**
   - The original Phase 6 LIDC ViT-Base specialist (`0.728` AUROC) was pretrained on the full LIDC corpus (transductive leakage of the LoRA evaluation patients), used a series-level pretraining split, and rested on a single seed (`+0.018` over the `0.710` 4-dataset ViT-Small baseline).
2. **What is already completed on disk (`2026-10-01` → `2026-10-02`):**
   - **Clean Patient-Held-Out Split (`data/mvp/split_manifest_lidc_patientholdout.json`):** `239` LoRA-eval patients (`malignancy_val` + `malignancy_test`, `243` series) held out of pretraining entirely; remaining corpus split by patient into `698` train series (`694` patients) and `77` val series (`77` patients). Verified `holdout ∩ pretrain == 0`.
   - **50,000-Step ViT-Base Pretraining (`runs/20261001_182138_lidc-specialist-vit-base-patient-holdout/`):** Completed on `game` (`RTX 4090`, job `2695fe2b1b26`), retaining all 10 checkpoints (`5K..50K`). Teacher entropy experienced a mid-run dip at step `30K` (`0.87`) before recovering (`4.70` at `45K`).
   - **Controlled View-Retrieval Sweep (`results/quality_view_retrieval_lidc_holdout.json`):** Evaluated all 10 checkpoints (`5K..50K`, layer 9, `N=512`) plus the Phase 6 50K control on the identical holdout split:
     - Pre-dip peak: **`holdout@20k`** (`ratio = 30.0x`, `top1 = 0.0586`, `top5 = 0.1699`, pre-collapse teacher entropy `5.04–7.76`).
     - Post-dip peak: **`holdout@40k`** (`ratio = 30.0x`, `top1 = 0.0586`, `top5 = 0.1953`).
     - Terminal: **`holdout@50k`** (`ratio = 28.0x`, `top1 = 0.0547`, `top5 = 0.1934`).
     - Control (`phase6-50k`): `ratio = 31.0x` (`top1 = 0.0605`, `top5 = 0.1875`).

## 2. Pre-Registered Gate

- **Checkpoint Selection (stated before LoRA runs):** Evaluate the pre-collapse peak **`checkpoint_00020000.pth` (`20K`)** and the post-recovery peak **`checkpoint_00040000.pth` (`40K`)** (plus `checkpoint_00050000.pth` as reference) using the standard LIDC malignancy LoRA protocol (`rank=8, alpha=16, 64px crops, lung HU window -30/120, 50 epochs, batch=32, lr=5e-4, early stopping on val AUROC`) across **3 seeds (`42, 43, 44`)** on the patient-held-out test set (`121` test patients, never seen by the backbone).
- **Verdict Rules:**
  - **PASS (`GO`):** 3-seed mean test AUROC `>= 0.720` **AND** 95% CI lower bound `> 0.710` (the 4-dataset ViT-Small reference baseline).
  - **PIVOT:** 3-seed mean test AUROC `>= 0.710` but 95% CI lower bound `<= 0.710` (within seed noise of the 4-dataset generalist).
  - **FAIL (`KILL`):** 3-seed mean test AUROC `< 0.710` (single-organ LIDC pretraining does not beat the 4-dataset generalist once transductive patient leakage is removed).

## 3. Atomic Ticket Breakdown (Option A — Split GPU Run from Evaluation/Verdict)

- **T1 (`Task` — Prep & Launch 3-Seed Patient-Holdout LIDC LoRA Benchmark via `bin/gpu_run`):**
  - Write `scripts/release/run_lidc_holdout_lora_3seed.py` (with unit tests in `tests/test_lidc_holdout_lora_3seed.py`) that enforces `assert_no_patient_leak` against `split_manifest_lidc_patientholdout.json`, runs `scripts/finetune_lora.py` across seeds `42, 43, 44` on `checkpoint_00020000.pth` (pre-collapse peak) and `checkpoint_00040000.pth` (post-recovery peak), and writes `results/lidc_patient_holdout_lora_3seed.json`.
  - Submit the job via `bin/gpu_run --id lidc-holdout-lora-3seed-v1 -- ...` and stamp `GPU job: lidc-holdout-lora-3seed-v1` onto Ticket T2.
- **T2 (`Task` — Adjudicate 3-Seed LIDC Patient-Holdout Verdict, Complete Adversarial Pass & Write `lawrenz.com` Brief):**
  - `Depends on: #<T1>`
  - `GPU job: lidc-holdout-lora-3seed-v1`
  - Read `results/lidc_patient_holdout_lora_3seed.json`, compute the 3-seed mean, standard error, and 95% CI on `malignancy_test`, and record the honest `GO / PIVOT / FAIL` verdict + 7-box Adversarial Pass in `docs/EXPERIMENTS_AND_RESULTS.md`, `docs/EXPERIMENT_TREE.md`, and `PROJECT_STATUS.md`.
  - Write `releases/launch-drafts/lidc-patient-holdout-brief.md` for `lawrenz.com` documenting the retraction → clean-split retrain → view-retrieval sweep → 3-seed LoRA verdict (whether positive or negative/null, per `publication-driver`).

# Retraction Notice — LIDC Single-Organ ViT-Base Specialist (0.728)

**Date:** 2026-09-20 · **Arm:** `lidc-specialist-vit-base-scale-aware` · **Verdict:** CONTAMINATED — RESULT RETRACTED

---

## What Was Claimed

Phase 6 trained a ViT-Base (86M) scale-aware specialist on LIDC-IDRI only
(160,620 slices), producing a LoRA malignancy AUROC of **0.728** — recorded as
the project's best malignancy result and marked "ready for HF Hub" in the
release tree. The specialist-vs-generalist claim rested on 0.728 vs a
ViT-Small 4-dataset baseline of 0.710.

## Why It Is Retracted

The 0.728 is **not a trustworthy generalization measurement**. Three independent
defects, none of which invalidate the *model* but all of which invalidate the
*number*:

1. **Transductive leakage.** The backbone was pretrained on the *entire* LIDC
   corpus including the patients later used for LoRA evaluation. The test
   patients' slices (unlabeled) were in the pretraining set — the same
   "dataset exhaustion" condition the hepatic-vessel card discloses as *not*
   generalization evidence. Here it was silently treated as evidence.

2. **Series-level (not patient-level) pretraining split.** `DATA_SOURCES.md`
   records "922 train series / 99 val series — same series-level split as
   5-dataset." LIDC-IDRI is 1018 patients / 981 series (~1:1 but not guaranteed).
   Any patient with 2+ series could span the train/val boundary. This is the
   exact leak class now guarded by `zoo/splits.py`; the Phase 6 run predates
   that guard.

3. **Razor-thin margin, no confidence interval.** 0.728 clears the 0.720 gate by
   0.008 — within seed noise — and there is no multi-seed estimate. The
   +0.018 delta over the 0.710 baseline cannot be distinguished from a lucky
   seed.

## Verdict Semantics

This is **not FAIL** (the hypothesis was never cleanly tested) and **not KILL**
(the arm is not dead — it is being re-opened under a corrected protocol). The
result is **retracted as a claim** and the specialist hypothesis returns to
**untested** status.

## What Survives

1. **The recipe is sound.** ViT-Base + scale-aware + LIDC-only training was
   stable (no collapse, 289 img/s, ~12.5h) — that part is reproducible and
   unaffected by the leakage.
2. **The label extractor was already correct.** `extract_lidc_malignancy.py`
   performs a patient-stratified train/val/test split. The LoRA-stage split was
   not the defect; the *pretraining* split was.
3. **The 0.710 baseline** (ViT-Small 4-dataset, patient-stratified LoRA) is
   unaffected and remains the reference to beat.

## What Changes Next Time

Re-run under a **patient-held-out pretraining split**:

1. Resolve `series_dir → PatientID` (via `zoo/splits.py`) for the full LIDC
   corpus.
2. Hold out ~20% of **patients** from pretraining entirely (the backbone never
   sees them).
3. Pretrain on the remaining patients; LoRA-eval only on the held-out patients.
4. Pre-register the gate (multi-seed AUROC with CI), and gate against the 0.710
   baseline with a margin that is not seed noise.

## Artifacts

- `runs/20260723_193825_lidc-specialist-vit-base-scale-aware/` — the (leak-contaminated) 50K run, retained for reference only
- `adapters/lidc-malignancy-vit-base-50k-lung-window/` — the 0.728 adapter; **retracted, do not cite as a result**
- `data/mvp/lidc_only_t2.csv`, `split_manifest_lidc.json` — series-level split inputs; superseded by the patient-held-out design

## Open Question

External validation for LIDC malignancy has no identified cross-institutional
held-out set (unlike CRLM for hepatic-vessel). Until one exists, the ceiling
for LIDC trust is a **clean internal patient-held-out** number — to be framed
as internal-only.

# Project Status — DINO-X

**Last updated:** 2026-10-01
**Phase / status:** Phase 10 ACTIVE — Release #2: LIDC clean-split retrain

## Current state

**Release #1 shipped.** The hepatic-vessel specialist
(`dinox-ct-msd-hepatic-vessel-vit-base-v1`) is public on the HF Hub with a full
evidence envelope, an external held-out validation on TCIA CRLM (slice AUROC
**0.9413**, 197 patients), and a published launch post. The release machinery is
proven end-to-end and reusable: `scripts/release/build_envelope.py`,
`releases/ENVELOPE_TEMPLATE.md`, `scripts/release/generate_data_provenance.py`.

**Release #2 is the LIDC specialist — and it is not a repackage.** The 0.728 result
was **retracted 2026-09-20** (transductive leakage: the backbone was pretrained on
the very patients later used for LoRA evaluation, plus a series-level pretraining
split). Release #2 therefore requires a **patient-held-out retrain** — the backbone
must never see the LoRA-eval patients. The corrected design is fixed in
`docs/RETRACTION_NOTICE_lidc-specialist-vit-base-scale-aware.md`.

The colon specialist remains `[CONCLUDED — FAIL]` (AUROC 0.6529) and is **blocked on
a new hypothesis** — per governance it is not launchable work.

## Immediate next action

1. Build a **patient-level LIDC split manifest** (`zoo/splits.py`: resolve
   series → PatientID, hold out ~20% of patients, `assert_no_patient_leak`) and
   **pre-register the multi-seed AUROC gate with a CI** before launching.
2. Launch the clean-split pretrain on **max395 via the GPU scheduler**
   (~12.5h, ViT-Base, recipe unchanged from Phase 6).
3. On completion: LoRA fine-tune on the train patients, evaluate **only** on the
   held-out patients, then run the release machinery (envelope → gates → HF).

## Headline result

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| Hepatic-vessel external slice AUROC (TCIA CRLM) | **0.9413** | ≥ 0.70 | **GO — RELEASED** |
| Hepatic-vessel Spearman (vessel-richness, score) | **0.527** (p≈1e-15) | — | GO |
| LIDC malignancy AUROC | 0.728 | ≥ 0.720 | **RETRACTED** — leakage; re-run under patient-held-out split |
| Colon specialist AUROC | 0.6529 | ≥ 0.70 | FAIL — new hypothesis required |
| ViT-Large pan-organ view retrieval | 34× (step 25K) | 100× | KILL — objective, not model size |
| Baseline (ViT-Small 4-dataset) | 0.710 | — | reference to beat |

## Next action (queue)

1. **Release #2 — LIDC clean-split retrain** (above).
2. **Colon** — new hypothesis needed before any re-run (organ-specific HU window,
   more patients, or a different task formulation).
3. **Tier 2 expansion** — pancreas-CT and CQ500 (data ready).
4. **Many-to-many view retrieval matching** — relax the 1-to-1 rule
   (arXiv:2604.23670).
5. **Cross-modality** — MRI (`dinox-mri-*`, UKBOB) and X-ray.

## Artifacts

- `releases/dinox-ct-msd-hepatic-vessel-vit-base-v1/` — Release #1 evidence envelope
- `releases/ENVELOPE_TEMPLATE.md` — reusable release machinery
- `docs/RETRACTION_NOTICE_lidc-specialist-vit-base-scale-aware.md` — Release #2 design
- `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md` — Phase 5 KILL artifact
- `docs/EXPERIMENTS_AND_RESULTS.md` — permanent ledger
- `docs/EXPERIMENT_TREE.md` — workstream map

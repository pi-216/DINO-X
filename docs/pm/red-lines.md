# Red Lines — DINO-X (Founder-Owned)

This file is owned exclusively by the founder (Tim). Agents may read it, never edit it.
Any suggestion or experiment arm that violates a rule below is ruled out forever.

1. **No Patient-Level or Series-Level Leakage Between Pretraining and Evaluation (`zoo/splits.py`):**
   Every pretraining and fine-tuning split must enforce `assert_no_patient_leak` and `assert_no_series_leak`. The retracted Phase 6 LIDC `0.728` result (`docs/RETRACTION_NOTICE_lidc-specialist-vit-base-scale-aware.md`) must never be cited as a valid generalization claim or packaged for release.
2. **No Pan-Organ Capacity-Dilution Scaling (`5dataset-phase5-large-bs256*`):**
   Discontinued permanently (`docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`). Scaling model size on the unstratified 5-dataset pan-organ DINO+KoLeo objective optimizes for scanner identity and spacing geometry rather than cross-organ anatomy.
3. **No Inline HU16 Formula Duplication:**
   `zoo/encoding.py` (`hu_to_u16` / `u16_to_hu`) is the single source of truth for HU16 encoding/decoding.
4. **No Public Model-Zoo Release Without the Full 5-Gate Evidence Envelope (`releases/ENVELOPE_TEMPLATE.md`):**
   A shipped base model requires cross-institutional external validation (like TCIA CRLM for `hepatic-vessel`). Internal patient-held-out evaluations (like LIDC when no external dataset is paired) must be framed honestly as internal-only until an external cohort is validated.
5. **No Single-Seed Release Verdicts:**
   Downstream LoRA specialist gates require multi-seed evaluation (minimum 3 seeds: `42, 43, 44`) with 95% confidence interval lower bound clearing the reference baseline (`0.710` for LIDC).
6. **No Bespoke Crons or Systemd Timers:**
   All work is scheduled via `ds4-queue` and bounded GPU runs via `bin/gpu_run` (`gpu-scheduler`).

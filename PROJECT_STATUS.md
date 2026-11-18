# Project Status — DINO-X

**Last updated:** 2026-11-18
**Governance:** `pi-216/DINO-X` — Board-driven under `founders-harness` (`AGENTS.md`, `bin/verify`). This file is a read-only orientation summary, never a work queue.

## Goal & Downstream Consumers

- **Public Medical Imaging Model Zoo (`Hugging Face Hub`):** Externally validated, scale-aware self-supervised CT specialists shipped with full 5-gate evidence envelopes (`moves:released-models`).
- **Public Research Output (`lawrenz.com`):** Method essays and honest post-mortems in `releases/launch-drafts/` feeding `pi-216/lawrenz.com` (**Project 12**).

## Headline Results (Ledger of Record)

| Model / Arm | Metric | Value | Gate | Verdict |
|---|---|---|---|---|
| **Hepatic-vessel v1** (`dinox-ct-msd-hepatic-vessel-vit-base-v1`) | External TCIA CRLM slice AUROC (`197` patients, `17,639` slices) | **0.9413** | ≥ 0.700 | **GO — RELEASED** (HF Hub + `lawrenz.com` essay live) |
| **3-slice vs 1-slice context** (`abl-3slice-context`) | External CRLM ΔAUROC (`[z-1,z,z+1]` vs `[z,z,z]`) | **−0.00017** | \|Δ\| ≤ 0.01 | **GO** — single-slice inference validated |
| **Frozen-backbone window probe** (`abl-frozen-probe`) | External CRLM linear probe (no LoRA) | **0.9068** | > null (`0.4467`) | **GO** — refutes window-artifact critique |
| **LIDC Phase 6 specialist** (`lidc-specialist-vit-base-scale-aware`) | Single-seed LoRA AUROC | 0.728 | ≥ 0.720 | **RETRACTED (2026-09-20)** — transductive patient leakage |
| **LIDC Phase 10 clean-split** (`lidc-specialist-vit-base-patient-holdout`) | 50K pretrain + layer-9 view retrieval (`20K` & `40K`: `30.0x` vs Phase 6 `31.0x`) | **30.0×** | 3-seed LoRA ≥ 0.720 & CI LB > 0.710 | **ACTIVE (`PRD-0001`)** — T1 launches 3-seed LoRA (`20K` & `40K`); T2 adjudicates verdict |
| **Colon specialist** (`colon-specialist-vit-base-scale-aware`) | Validation AUROC | 0.6529 | ≥ 0.700 | **FAIL** — requires new hypothesis from `research-agent` |
| **ViT-Large 5-dataset pan-organ** (`5dataset-phase5-large-bs256-v2`) | View retrieval @ 50K | 34× | ≥ 100× | **KILL** — capacity dilution / scanner fingerprinting |

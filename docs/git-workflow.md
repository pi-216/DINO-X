# Git Workflow — Branch-per-Model + Merge-Back

This is the standing version-control policy for DINO-X. It exists to answer one
question reliably: **is this change a durable asset or an experimental byproduct?**

The recurring failure this prevents is the "unidentifiable dirt" pile — uncommitted
adapter weights, scratch CSVs, and deleted run artifacts sitting in the working tree
with no clear owner or home. When a repo can't distinguish reusable code from
one-model experiment state, both get mixed and neither is trusted.

---

## The Two Kinds of Change

| Kind | Examples | Lives in |
|---|---|---|
| **Core asset** (survives the model) | `zoo/` modules, shared `scripts/`, `tests/`, `docs/`, governance, skills | `main`, always |
| **Model artifact** (ephemeral) | adapter weights, run configs, checkpoints, scratch metrics | `model/<slug>` branch, then NAS / HF Hub — *not* git |

Everything in the working tree should be classifiable into exactly one of these. If
a file fits neither, it is scratch and belongs in `.gitignore` or a deleted-temp.

---

## Branch Strategy

### `main` — the durable core

Contains only core assets. Always in a runnable, documented, tested state. A fresh
`git clone` of `main` gives a working toolkit with no model-specific baggage.

### `model/<slug>` — one branch per model or experiment arm

All the churn for one model lives here: preprocessing tweaks, ablation configs, LoRA
adapter weights, model-card drafts, eval exploration. These branches may be long-lived
while a model is active and can be deleted after release + merge-back.

Arm slugs follow the governance convention (lowercase-hyphen, e.g.
`lidc-malignancy-lora-r8-64px-lung-window`), so branch name = arm slug.

---

## Merge-Back Rule (the discipline)

A branch merges back to `main` **only the deltas that are not model-specific**. The
branch is never merged wholesale; individual reusable changes are brought over.

**Merges back to `main`:**
- New or fixed `zoo/` code (e.g. `zoo/encoding.py`, `zoo/splits.py` — the canonical
  encoding + patient-split modules)
- Shared script improvements (`finetune_lora.py`, `evaluate_panorgan.py`, etc.)
- New tests
- Documentation: ledger verdicts, `EXPERIMENT_TREE.md` updates, research notes,
  `PROJECT_STATUS.md`
- Governance: `AGENTS.md`, this file, experiment-structure rules
- Skill updates (skills live in the Hermes profile, but a repo pointer/note may belong)

**Stays on the branch / goes to NAS+HF Hub (never `main`):**
- Adapter weights (`*.safetensors`, `*.pth`)
- Per-run configs and checkpoints
- `tunableop_results*.csv` and other scratch metrics
- Model-card *binaries* (the card *source* / evidence envelope text belongs in git)

The test for a change: **"if I deleted this model entirely, would this still be worth
keeping?"** If yes → merge back. If no → it's model-specific.

---

## Artifact Routing

| Path | Home |
|---|---|
| `runs/` | symlink to NAS — not in git (see `.gitignore`) |
| `adapters/` | NAS + HF Hub — gitignored; keep only a pointer/index if needed |
| `releases/` | evidence-envelope **text** in git; binaries gitignored |
| `data/`, `*.dcm`, `*.nii`, `*.png` (under data/) | never in git |
| scratch at repo root (e.g. `tunableop_results*.csv`) | gitignored |

Medical data is never committed. Raw imaging and derived datasets live on NAS and HF
Hub only.

---

## Working-Tree Hygiene

- Before starting a model, branch from `main` and record `git_commit` (already a
  Critical Rule — the SHA is the reproducibility anchor).
- Before finishing a session, either commit core work to `main` or leave it clearly
  staged on the model branch with an intent note.
- Untracked files are a bug signal. If you see `??` entries you can't classify, stop
  and route them — don't leave them for the next person to wonder about.

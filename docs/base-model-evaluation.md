# Base-Model Evaluation Protocol — DINO-X

> **Status: PROPOSAL — under review. Not yet a standing rule.**
> Companion to `data-selection.md` (which governs *what data feeds the model*).
> This governs the question that data-selection deliberately left open: **how do
> you prove a self-supervised base model is actually good?**

---

## 0. Why this doc exists

The zoo's thesis is:

> **The base model is the product. The LoRA adapters are probes, not products.**
> A well-trained self-supervised CT representation that *scientific progress can
> be built on top of*.

That goal is deliberately "squishy" only if you try to score it with one number
on one task. The self-supervised learning field already solved this exact problem
(DINO, SimCLR, MoCo): you do **not** evaluate a base model by "how good is it at
X" — you evaluate it by **how much it helps across a battery of downstream probes,
measured as a delta over a baseline ladder.**

This doc turns that principle into a fixed, versioned, reproducible protocol.

---

## 1. The governing principle

> **A base model's quality is its *transfer utility* — the delta it provides over
> a baseline ladder, across a fixed probe battery, on clean patient-level splits.**

Two corollaries that this principle settles:

1. **No single probe is the model.** One good LoRA AUROC proves one probe, not the
   base model. A base-model claim requires the *whole battery* against the *whole
   ladder*.

2. **Absolute numbers are not the claim; the delta is.** "AUROC 0.71" is not a
   base-model claim. "Outperforms every baseline on N of M probes by a median Δ of
   X" is.

---

## 2. What a probe IS — and is NOT

This distinction is the load-bearing one, and it corrects a category error this
project has made (and that caused the Duke chase to go sideways):

| | Probe | Release |
|---|---|---|
| **Purpose** | Demonstrate the base model is useful | A model someone can rely on |
| **What it needs** | A **clean patient-level split** | Clean split **+ external validation** |
| **Contamination sin** | Leakage (the LIDC retraction) | Leakage **and** absent external set |
| **Governance** | This doc | `data-selection.md` external floor |

**A proof-LoRA does not need external validation.** It needs a clean split. The
LIDC malignancy LoRA's sin was the leak (transductive pretraining on the full
LIDC corpus + series-level split), *not* the absence of an external set. External
validation is required **only when a model is shipped** (per `data-selection.md`
§6, now a standing project standard).

---

## 3. The probe battery

A small, **fixed**, versioned set of downstream heads, each reproduced against
every new base model. The battery must mix label-free and labeled probes so no
single failure mode (collapse, spacing mis-encoding, organ dilution) goes unseen.

**Candidate battery** (drawn from machinery already in the repo — this is a
*consolidation*, not a new invention):

| Probe | Source | Signal it guards |
|---|---|---|
| `view_retrieval` (label-free) | `scripts/phase5_view_retrieval_eval.py` | Representation collapse / instance discrimination |
| `linear_probe_malignancy` | frozen-backbone linear head on LIDC | Downstream classification transfer |
| `spacing_r2` | scale-embedding eval | Is physical scale actually encoded? |
| `dataset_discrimination_auc` | cross-dataset eval | Is the model learning dataset idiosyncrasies vs. anatomy? |
| `organ_cluster_purity` | centroid eval | Cross-organ structure |

The battery is **versioned** — adding or changing a probe bumps the protocol
version, so a base model is always compared under the same battery it was judged
under.

---

## 4. The baseline ladder

A base model's score is its **position on this ladder**, not any single number.
The ladder is fixed and reproduced on the *same* probe battery:

| Rung | Model | Role |
|---|---|---|
| 0 | Random-init backbone | Floor — what does no training give you? |
| 1 | Supervised ImageNet-pretrained ViT | Generic-transfer baseline |
| 2 | Prior DINO-X base model | Regression guard — is the new one actually better? |
| 3 | **Candidate base model** | The model under evaluation |

A base-model claim reads: *"The candidate sits at rung 3 and clears rung 2 by a
median Δ of X across the battery, with no probe below rung 1."*

---

## 5. The claim template

When a base model is evaluated, the result is recorded in the ledger using this
template (mirrors the existing arm format, extended with the ladder):

```
**Goal:** Base-model transfer utility of {candidate}.
**Battery:** {probe list, protocol version}.
**Ladder:** {baseline results}.
**Delta:** {candidate vs. each rung, per probe + median}.
**Clean-split attestation:** {patient-level split confirmed, ref to zoo/splits.py}.
**Verdict:** {GO / GO-with-caveat / KILL} + adversarial pass checklist.
```

---

## 6. Open questions (genuine, not literature-settleable)

1. **Probe selection.** Is the 5-probe candidate battery above the right one, or
   does it need a segmentation probe (e.g. a frozen-feature linear-seg on MSD)
   before it can claim "cross-anatomy"?
2. **The ImageNet baseline.** Rung 1 assumes ImageNet-supervised ViT is a fair
   generic baseline. For medical CT this is contested — the ladder may need a
   domain-matched baseline instead.
3. **Delta thresholds.** How large a Δ counts as "meaningfully better"? This is
   the one place a pre-registered number is genuinely useful, and it is not yet
   set.

---

## 7. Relation to the rest of the governance

- **`data-selection.md`** governs *what goes in* and *when a model may ship*.
- **This doc** governs *how a base model is judged to be good*.
- **`zoo/splits.py` + `zoo/encoding.py`** are the *enforcement* layer both depend
  on (clean splits, canonical encoding).
- **`docs/git-workflow.md`** governs where each model's churn lives.

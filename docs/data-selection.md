# Data Selection Governance — DINO-X

> **Status: PROPOSAL — under review. Not yet a standing rule.**
> This is the missing layer between the *operational* data docs (`data_fuel.md`
> — where data lives; `data_preprocessing.md` — how it's encoded; `DATA_SOURCES.md`
> — what exists) and the *experiment* governance (`experiment-structure.md`).
> It answers the question none of them do: **what data *should* we use, how much,
> and what's the quality bar?**
>
> §3–§6 are now grounded in a literature review (2026-09-20, three parallel
> research passes). Citations are inline; where the literature has **no**
> established number, that is stated explicitly rather than filled with a guess.

---

## 0. The Problem This Solves

Every experiment in the ledger pre-registers its *model* gate (AUROC, view
retrieval, entropy) but nothing pre-registers the *data* gate. Data choice has
been a background assumption — "use LIDC because it's lung," "more slices is
better" — never a falsifiable decision with a written quality bar.

The LIDC retraction is the cost of that gap: a model was trained on a corpus
whose *split* was never validated (series-level on a multi-series-per-patient
dataset) and whose *size* was assumed sufficient. The model failed not on
biology but on data governance.

This document makes data selection a first-class, gated decision.

---

## 1. Eligibility Criteria (what qualifies)

A dataset may enter a training index only if ALL hold:

| # | Criterion | Check |
|---|---|---|
| 1 | **Anatomical/task match** — organ and task align with the model's scope | Stated in the arm's `provenance.yaml` hypothesis |
| 2 | **Modality match** — CT for CT specialists, etc. (no silent cross-modality mixing) | Dataset YAML `modality` field |
| 3 | **License + redistribution** — CC-BY / CC-BY-SA usable; NC/ND is trainable-but-not-redistributable and must be flagged | `zoo/datasets/ct/*.yaml` license field |
| 4 | **Scanner/protocol diversity** — if the goal is cross-institution generalization, the corpus must span >1 institution OR the arm must explicitly pre-register single-institution scope | Documented in `provenance.yaml` |
| 5 | **Identity resolvable to patient** — `series_dir → PatientID` resolvable (via `zoo/splits.py`), because patient-level splits are mandatory (§4) | `read_patient_id` returns non-None on spot check |

**Rule:** a dataset that fails #5 (identity can't be resolved to patient) is
**trainable only for exploratory arms**, never for a confirmatory arm whose gate
is a generalization claim.

---

## 2. Provenance Requirements (before training)

Every dataset entering a training index must carry, frozen at run start:

- **Source** (institution, collection, DOI/citation)
- **License + redistribution status**
- **Exact slice/series/patient counts** (measured, not recalled — reconcile to `index.csv`)
- **The split's identity granularity** (`patient_id` vs `series_dir`) — mandatory; `series_dir`-only splits are a red flag
- **A data manifest hash** (already produced by `zoo/lineage.py`)

This is not new — `DATA_SOURCES.md` + `zoo/lineage.py` already do most of it. The
gap is that **none of it was gated**: nothing *refused* to train when a field was
missing. This section turns "documented" into "enforced."

---

## 3. Quality Gates (written, measurable)

Each gate is script-checkable, not judgment-based. Confidence labels:

- **established** — a number exists in the literature / reference implementation
- **common-practice** — widely used, not formally validated
- **needs-calibration** — no published threshold; must be measured on DINO-X's own data

### 3.1 Established gates (adopt as-is)

| Gate | Threshold | Source |
|---|---|---|
| **Patient-level split** | 0 shared patients across folds; split key = resolved PatientID, not series/study/accession | Rouzrokh et al. 2022, Radiology:AI; Wen et al. (slice-level CV inflates accuracy 30–55%) |
| **NaN/Inf check** | reject any case with NaN/Inf voxel | nnU-Net `verify_dataset_integrity.py` |
| **Image/label geometry match** | segmentation must match image shape + spacing | nnU-Net `verify_dataset_integrity.py` |
| **Label value validity** | consecutive ints from 0 (background), no out-of-vocab | nnU-Net `dataset_format` docs |
| **Inter-rater agreement (categorical)** | Fleiss/Cohen kappa ≥ 0.61 ("substantial"); <0.60 flag for review | Landis & Koch 1977 |
| **Inter-rater agreement (continuous)** | ICC ≥ 0.75 ("good"), report 95% CI | Koo & Li 2016 |
| **HU reference ranges** | air −1000, water 0, bone +700…+1000, metal >2000; calibration anchor water=0/air=−1000 | Hounsfield scale; Boas & Fleischmann |

### 3.2 Common-practice gates (adopt, expect to tune)

| Gate | Threshold | Source |
|---|---|---|
| **Intensity normalization** | clip 0.5–99.5 percentile, subtract foreground mean / std | nnU-Net CT scheme |
| **Spacing resampling** | resample to dataset median voxel spacing per axis | nnU-Net / TotalSegmentator |
| **Empty/all-air detection** | flag implausibly small foreground volume | nnU-Net `verify_labels` |
| **Metal-artifact tracking** | flag >~2000–3000 HU + streaks (~21% incidence — track, don't blanket-exclude) | Boas & Fleischmann |

### 3.3 Needs calibration (DINO-X must measure, not assume)

No published number exists for these — the threshold must be calibrated on our
own DICOMs before it becomes a standing rule:

1. **Z-gap / slice-spacing discontinuity** — "how irregular before a series is corrupt." Calibrate a tolerance like `median(z-gap) + k·MAD`.
2. **All-air vs all-bone HU cutoff** — the exact percentile/volume fraction that flags a scan.
3. **Metal-exclusion severity** — whether to exclude vs keep-and-flag, and at what extent.
4. **Anisotropy ceiling** — max acceptable in-plane:z spacing ratio.
5. **Minimum rater count** for a stable kappa/ICC (≥3 raters, but item count per task).

---

## 4. The "More Data" Hypothesis — settled in principle

The standing intuition "more data is better" is **falsified in the existing ledger**
and now also by the literature. It is a *testable claim*, never an assumption:

| Ledger evidence | Finding |
|---|---|
| 5-dataset (386K slices) vs 4-dataset (296K) | AUROC **0.680 vs 0.710** — *more* data, *worse* result |
| ViT-Large capacity dilution | **KILL** — more params + data collapsed (colon↔vessel cosine 0.962) |
| Crop size | 64px >> 128px >> 224px — *bigger* input was *worse* |
| Varoquaux & Cheplygina 2022 | larger cohorts did NOT improve AD diagnosis over a decade |
| Northcutt et al. 2021 | even ~5–10% label noise degrades models; *more* noisy data doesn't help |
| Xray-Visual (arXiv 2602.16918) | semantic resampling beats raw scale; noise filtering > more data |

**The established rule:** grow quantity only **within the target distribution and
with audited label quality**. More data *harms* when it injects label noise,
distribution shift, or off-domain examples — not from raw volume. "Enough" is
answered by a **learning curve + confidence interval**, not by defaulting to
"use everything."

### Metric discipline (settled, and it's not optional)

- **Count patients, not slices** — a patient contributes many slices and inflates apparent N (Varoquaux & Cheplygina 2022).
- **Report AUROC *and* AUPRC**, and compare AUPRC against its prevalence baseline `P/(P+N)`, never 0.5 (Saito & Rehmsmeier 2015; McDermott et al. 2024).
- **Drop accuracy as a headline metric** for binary tasks — it breaks under imbalance. Report AUROC/AUPRC + sensitivity/specificity at a fixed point (Buda et al. 2018).

---

## 5. The Validation Question — the deepest one, now answerable

The four open questions from the draft are now resolved. The governing principle
the literature converges on: **the strength of your validation must match the
strength of your claim** — not "external validation is always required."

**Answer to "is internal-only ever sufficient?":** Yes — for a *non-clinical,
research-use* model release, honest internal-only disclosure is an accepted
baseline, **provided the claim matches the evidence**. What is *not* acceptable
is releasing with language implying generalization or clinical readiness when
only a held-out internal split was done. The RSNA 2022 systematic review is the
citation: 81% of externally-validated radiology models degraded on external data
(49% by ≥0.05 AUC). Internal numbers routinely overstate real-world performance.

**The rule, concretely:**

| Claim you want to make | Validation required |
|---|---|
| "performs on this institution's population" | internal patient-held-out split (necessary, not sufficient) |
| "generalizes to other scanners/sites" | **external** (cross-institution) — de facto mandatory |
| "clinically deployable" | external + multi-site (~3 sites) + FDA independence standards |

**Vocabulary (TRIPOD, use it in every card):**
- Type 2a — internal random split (what LIDC *should* have had)
- Type 2b — internal non-random / leave-one-site-out (stronger, still not external)
- Type 3 — external, different institution (what hepatic-vessel got via CRLM)

**Direct consequence for LIDC:** release is permissible *if and only if* the card
says "internally validated on a held-out patient split; no external validation
performed; generalization to other scanners/institutions is unproven." It may
*not* say "validated" or "generalizes."

**Project standard (decided 2026-09-20):** the DINO-X zoo holds **every** model
to the external-validation bar before release — not merely the literature's
permissible internal-only baseline. A model without cross-institution external
validation is **not release-eligible**, regardless of how strong its internal
held-out number is. This is a stricter floor than the reporting standards
require, adopted deliberately because the zoo's credibility is its differentiator
and hepatic-vessel already set the precedent. Consequence: LIDC is
**not-yet-releasable** until it has an external held-out set (currently an open
gap — see below).

**Reporting standard:** model cards should target **TRIPOD+AI (BMJ 2024)** and
**CLAIM 2024** — both require disclosing validation type and performance with
confidence intervals, which is exactly what defends an internal-only release.

---

## 6. How This Becomes Enforced (the roadmap)

This doc alone changes nothing — it's a proposal. Enforcement requires three
concrete artifacts, in order:

1. **A data-gate checklist script** — extend `phase4_make_split_manifest.py` or
   add `zoo/data_gate.py` that runs §1–§3 checks and refuses to emit a training
   index on failure. The §3.1 gates are already mechanical (nnU-Net has reference
   implementations); the §3.3 gates need calibration data first.

2. **Calibrate §3.3 thresholds** — measure z-gap/all-air/metal/anisotropy on
   DINO-X's own corpora (exploratory), then re-register as confirmatory gates.

3. **A pre-registered scaling experiment** — confirmatory arm testing "how much
   data is enough" on the next specialist (e.g. LIDC retrain at N = 25% / 50% /
   100% of patients), producing the learning curve that replaces "more data"
   intuition.

### Remaining genuine open questions (post-review)

These are *decisions*, not facts the literature can settle:

1. **Which calibration to do first** — my recommendation: the split/contamination
   gate (already settled, just needs wiring) and the z-gap gate (the LIDC-class
   failure), before all-air/metal.
2. **What external held-out set to use for LIDC** — the standard is now settled
   (external required before release), but *which* cross-institution lung-malignancy
   dataset serves as LIDC's CRLM-equivalent is unresolved. This is the blocking
   gap for LIDC's release; it needs research, not just a decision.

---

## Relationship to Existing Docs

| Doc | Role | This doc's relationship |
|---|---|---|
| `data_fuel.md` | where data lives, acquisition runbook | operational — untouched |
| `data_preprocessing.md` | how data is encoded (HU16, windowing) | operational — untouched |
| `DATA_SOURCES.md` | what datasets exist + provenance catalog | **inputs** to §1–§2 |
| `experiment-structure.md` | how experiments are gated | **extends** — data is now gated too |
| `zoo/splits.py` | patient-identity + split enforcement | **the enforcement mechanism** for §1#5, §3 |

## Key References

- TRIPOD (Moons et al. 2015) / TRIPOD+AI (Collins et al. BMJ 2024) / CLAIM 2024 (Tejani et al.)
- Yu AC et al., "External Validation of Deep Learning Algorithms for Radiologic Diagnosis," *Radiology:AI* 2022;4(3):e210064 (81% external degradation)
- Varoquaux & Cheplygina, *npj Digit Med* 2022;5:48 (patient-vs-image counting; cohort size futility)
- Saito & Rehmsmeier 2015; McDermott et al. 2024 (AUROC/AUPRC under imbalance)
- Landis & Koch 1977; Koo & Li 2016 (kappa / ICC bands)
- nnU-Net `verify_dataset_integrity.py` + `dataset_format.md` (mechanical integrity gates)
- FDA/Health Canada/MHRA "Good Machine Learning Practice" (Oct 2021) (independence + representativeness)
- Northcutt et al. 2021 (label noise); Rouzrokh et al. 2022 (data-handling bias)

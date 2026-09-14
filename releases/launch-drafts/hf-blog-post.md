---
title: "DINO-X: a CT organ-specialist that ships the evidence, not just the weights"
slug: dinox-ct-organ-specialist-evidence-envelope
summary: "A scale-aware ViT-Base for hepatic-vessel detection, external-validated at 0.9413 — released with a full evidence envelope: the negative results, the leakage audit, and a recompute script."
tags: ["medical-imaging", "ct", "vision-transformer", "self-supervised", "model-release"]
---

# DINO-X: a CT organ-specialist that ships the evidence, not just the weights

Most model releases give you a checkpoint and a number and ask you to trust both.
This one is the first of a small CT model zoo (DINO-X), and it ships with a full
**evidence envelope**: the negative results, the leakage audit, and a script that
recomputes the headline metric. You shouldn't have to trust us — and here, you don't.

## The model

<iframe
	src="https://huggingface.co/timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1"
	frameborder="0"
	width="100%"
	height="450"
></iframe>

**`dinox-ct-msd-hepatic-vessel-vit-base-v1`** — a scale-aware ViT-Base (86M) organ
specialist for hepatic/portal vessel tissue detection in abdominal CT.

- **Backbone:** ViT-Base with a `ScaleEmbedding` MLP that ingests DICOM spacing
  `(spacing_x, spacing_y, slice_thickness)` — self-supervised (DINO) on a single
  organ dataset (MSD Hepatic-Vessel, Task 08).
- **Head:** LoRA (rank 8) + a 2-class linear head, **merged into the backbone at
  release** so inference needs no `peft`.
- **Headline metric:** external slice AUROC **0.9413** on TCIA CRLM (197 patients,
  17,639 slices — a genuinely disjoint institution/scanner set).

## One-liner inference

```python
from zoo.predict import load_classifier, predict_proba

clf = load_classifier("timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1")
p = predict_proba(clf, hu_slice, pixel_spacing=(0.77, 0.77), slice_thickness=1.5)
```

The `zoo.predict` helper ships with the DINO-X codebase. Install it straight from
GitHub (it is not on PyPI — do not `pip install dinox-zoo`, that name is not
published):

```bash
git clone https://github.com/timlawrenz/DINO-X.git
cd DINO-X
pip install torch torchvision            # build for your hardware (CPU/CUDA/ROCm)
pip install -r requirements-inference.txt
pip install .                            # makes `from zoo.predict import ...` work
```

The trained −30/120 HU window and the spacing embedding are baked in, so
preprocessing can't silently diverge from the validated pipeline. The merged
backbone makes the LoRA double-wrap footgun structurally impossible.

## The evidence envelope (what "release" means here)

Every zoo model ships five things beyond the weights:

1. **Data provenance** — sources, licenses, splits, slice counts, machine-generated
   from the on-disk files (`data_catalog.md` / `data_provenance.json`).
2. **The decision trail** — the pan-organ failure that pushed us to organ
   specialists, including a documented **KILL** with a discontinuation notice: a
   multi-organ generalized model that learned scanner identity over anatomy and
   collapsed (view retrieval 34× vs a 100× gate, scanner-fingerprinting AUC 0.981).
3. **External validation** — the TCIA CRLM held-out evaluation (0.9413).
4. **Negative results, published as assets** — e.g. the colon specialist that
   FAILed its gate (AUROC 0.6529) and was *not* released.
5. **A recompute script** — `reproduce/repro_metrics.py` recomputes the external
   number from persisted predictions.

## The adversarial process (what survived review)

This release went through multiple adversarial review rounds plus an internal
red-team. The findings that survived: disclosing them is the credibility.

- **Dataset exhaustion (not just leakage).** MSD Task08 has exactly 303 series, and
  both pretraining and fine-tuning used all of them. So every *internal* metric is
  **disclosure-only, not generalization evidence** — and we label it exactly that
  in `evaluation.json`. The external CRLM number is the sole release-supporting
  metric.
- **A degenerate metric we removed.** Patient-level ROC-AUC on CRLM is undefined
  (196/197 patients vessel-positive, 1 all-negative), so we don't report it. The
  Spearman rank correlation (0.527, p≈1.8e-15) is the valid patient-level signal.
- **A window-domain-shift claim, tested and rejected.** A reviewer argued the
  narrow −30/120 fine-tune window is a "massive domain shift" the frozen backbone
  can't interpret. We tested it: the **frozen backbone, no LoRA, hits AUROC 0.9068**
  under the narrow window. LoRA then lifts it to 0.9413. Not a preprocessing
  artifact — measured, not asserted.
- **A 2.5D-vs-2D ablation.** The classifier duplicates a single slice across 3
  channels. A reviewer asked how much that costs. We measured it on all 17,639
  external slices: true 3-slice context 0.94113 vs duplicated 0.94130 — **Δ =
  −0.0002**. Collapsing the z-axis costs nothing measurable here.

## Scope and limits (read this before using it)

- **Research use only.** Not a clinical device. Raw, uncalibrated softmax
  probabilities.
- **Scale range.** The scale embedding was trained on spacing_x/y 0.57–0.98 mm and
  slice thickness 0.8–8.0 mm. Outside that manifold is out-of-distribution — the
  model returns a value but correctness isn't validated there.
- **Proof of representation.** The headline value is the **pretrained backbone**
  for downstream fine-tuning; the vessel classifier is the demonstration that it
  learned real signal.

## The zoo ahead

This is model #1. The process — pre-registered gates, an adversarial pass, an
evidence envelope, a recompute script — is the repeatable part. The LIDC malignancy
specialist (AUROC 0.728) is next up; where it lands depends on this same envelope.

**Repo:** [`timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1`](https://huggingface.co/timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1) ·
**Code:** [github.com/timlawrenz/DINO-X](https://github.com/timlawrenz/DINO-X)

*We don't ask you to trust the number. We give you the failures, the audit, and
the script.*

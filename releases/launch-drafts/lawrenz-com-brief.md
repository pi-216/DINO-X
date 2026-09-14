# Brief — lawrenz.com post: "A model whose numbers you don't have to trust"

**Target repo:** `/home/tim/source/activity/lawrenz.com` (`_posts/2026-09-14-dinox-model-zoo-trust.md`)
**Profile:** `lawrenz.com` hermes profile (Jekyll + jekyll-scholar)
**Companion piece:** HF blog post (technical launch) — this essay is the *method* behind it.
**Hero claim:** the release's value is a *process* (gates → red-team → measure-don't-speculate → evidence envelope), not a single AUROC.

---

## Framing (the angle that makes it worth reading)

Most ML releases hand you a checkpoint and one number and ask for faith. This post is the
story of shipping a CT model **deliberately** — and of discovering, repeatedly, that an
external reviewer's "that can't be right" claim was answered best by *measuring it*, not
editing the docs.

---

## Part 1 — The model zoo problem (and why one model isn't a zoo)

- Hook: the imposter fear — "is a single model a zoo, or a pet?" — resolved by making the
  *process* the unit of credibility, not the count of models.
- Setup: DINO-X, scale-aware CT organ specialists; the pan-organ failure (a `KILL` with a
  discontinuation notice) that made specialists the right call.
- Evidence: the first release is a hepatic-vessel specialist, external AUROC 0.9413.
- Turn: a model zoo isn't "many models" — it's a *trustworthy pipeline for models*.
- Meaning: model #2 is cheap because #1 defined the process.
- Handoff: → what makes a release trustworthy (the envelope).

## Part 2 — The evidence envelope (the five things beyond the weights)

- Hook: a release is a decision trail, not a binary.
- Setup: five deliverables — provenance, decision trail, external validation, negative
  results, recompute script.
- Evidence: each maps to a file (`data_catalog.md`, `the_science/*`. `reproduce/repro_metrics.py`).
- Turn: negative results are credibility assets — the colon FAIL (0.6529) is published
  with the same rigor as the successes, not buried.
- Meaning: you shouldn't have to trust; the envelope is the mechanism.
- Handoff: → but the envelope only works if the numbers survive attack.

## Part 3 — Adversarial review as a feature, not a threat

- Hook: invite the attack; the docs disarm it.
- Setup: four review rounds + internal red-team before flipping public.
- Evidence (parallel to the six-beat arcs below): each reviewer claim got measured, not edited.
- Turn: the well of *valid* criticism runs dry fast when you answer falsifiable claims with experiments.
- Meaning: "reviewer-proof" is a process property, not luck.
- Handoff: → the four falsifiable claims and the numbers that closed them.

## Part 4 — Four claims, four numbers (the measurable-heart of the essay)

Four reviewer attacks, each answered by an experiment (numbers verified against artifacts):

1. **"The narrow −30/120 window is a massive domain shift."** → frozen backbone, no LoRA:
   AUROC **0.9068** (LoRA → 0.9413). Window-robust. `{: .tip }` "measure, don't speculate."
2. **"Duplicating a single slice across 3 channels costs accuracy."** → duplicated vs true
   3-slice context: **Δ = −0.0002** (17,639 slices). Costs nothing. Validated design choice.
3. **"Dataset exhaustion makes internal metrics meaningless."** → true, so internal number
   is labeled `DISCLOSURE_ONLY_not_a_metric`; external number is the sole evidence. Framing, not evasion.
4. **"Patient-level AUROC 0.739 is degenerate (196:1)."** → true; **removed** as a metric;
   Spearman 0.527 is the valid patient-level signal.

- Meaning: falsifiable claims deserve numbers, not prose.
- Handoff: → so what does "trust" look like structurally (the envelope as the trust object).

## Part 5 — The scale embedding and its operating range

- Hook: "scale-aware" is a promise; define where it holds.
- Setup: `ScaleEmbedding` MLP ingests DICOM spacing; training manifold spacing_x/y 0.57–0.98 mm,
  thickness 0.8–8.0 mm.
- Evidence: outside that manifold is OOD — the model returns a value, correctness unvalidated.
- Turn: defining the boundary protects the model (garbage-in ≠ broken model).
- Meaning: an honest OOD map is part of release quality.
- Handoff: → the zoo that comes next.

## Part 6 — The zoo is a pipeline, the model is a receipt

- Hook: the classifier is proof the backbone learned something; the backbone is the product.
- Setup: merged LoRA (no `peft`), one-liner inference, `pip install dinox-zoo`.
- Evidence: merged == unmerged prediction verified (0.107753).
- Turn: "proof of representation" framing aligns the card with reality.
- Meaning: ship the backbone; the classifier is the receipt.
- Handoff: → LIDC malignancy (0.728) is next, gated by this same envelope.

## Numbers at a Glance (summary table)

| Metric | Value | Status |
|---|---|---|
| External slice AUROC (TCIA CRLM) | 0.9413 | release-supporting |
| External Spearman (patient signal) | 0.527 (p≈1.8e-15) | valid |
| Internal AUROC (MSD) | 0.9456 | disclosure-only (dataset exhaustion) |
| Frozen backbone probe (no LoRA) | 0.9068 | refutes window-shift claim |
| 3-slice vs duplicated Δ | −0.0002 | validated single-slice path |
| Colon specialist | 0.6529 | FAIL, not released |

## References (new BibTeX to add to `_bibliography/references.bib`)

- `antonelli2022msd` (MSD, NatComm 10.1038/s41467-022-30695-9)
- `simpson2023crlm` (TCIA CRLM, 10.7937/QXK2-QG03)
- self-cite the HF model repo (web link, as misc)

## Tags

`research`, `dino-x`, `model-zoo`, `ct`, `self-supervised`, `release`

## Verification checklist (before publish)

- [ ] Every number matches the ledger / `evaluation.json` (0.9413, 0.9068, −0.0002, 0.527, 0.9456, 0.6529).
- [ ] `{% cite %}` for MSD + CRLM; `{% bibliography --cited %}` at end.
- [ ] TOC after intro; `{: .tip }` for the "measure, don't speculate" lesson.
- [ ] Cross-link to any prior DINO-X/self-supervised post if one exists.
- [ ] Thumbnail: pick strongest figure (the external-eval or the frozen-probe table).
- [ ] `bundle exec jekyll build` clean; verify citations render APA.

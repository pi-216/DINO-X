# Brief — lawrenz.com post: "We retracted our best LIDC number, retrained honestly, and lost"

**Target repo:** `/home/tim/source/activity/lawrenz.com` (`_posts/2026-11-XX-dinox-lidc-patient-holdout.md`)
**Profile:** `lawrenz.com` hermes profile (Jekyll + jekyll-scholar)
**Companion piece:** none — this is a standalone negative-result post in the DINO-X series.
**Hero claim:** a clean measurement that says "no" is worth more than a contaminated one that says "yes" — and the machinery that catches your own leak is the actual deliverable.
**Framing constraint (red line #4):** everything below is an **internal** patient-held-out evaluation. LIDC has no paired cross-institutional cohort; no release is claimed, and the word "generalization" is reserved for externally validated results only.

---

## Framing (the angle that makes a negative result worth reading)

In September we retracted our headline malignancy number (0.728) because the model had been
pretrained on the very patients it was later scored on. In November we finished the do-over:
same recipe, same budget, one difference — the eval patients never entered pretraining. The
answer: **0.692 / 0.689 vs a 0.710 baseline. The specialist was the leak.** This post is the
full technical record — split, trajectory, checkpoint sweep, three-seed gate — published with
the same rigor as our wins.

---

## Part 1 — How the 0.728 leak was caught and retracted

- The Phase 6 LIDC ViT-Base specialist scored **0.728** LoRA AUROC, cleared its 0.720 gate, and
  was marked "ready for HF Hub."
- Three independent defects killed it (`docs/RETRACTION_NOTICE_lidc-specialist-vit-base-scale-aware.md`, 2026-09-20):
  1. **Transductive leakage:** the self-supervised pretraining set was the *entire* LIDC corpus —
     the LoRA test patients' unlabeled slices were in it. The hepatic-vessel model card already
     discloses this condition as *not* generalization evidence; we had silently treated it as evidence.
  2. **Series-level pretraining split:** train/val were split by series, not patient; any patient
     with 2+ series could straddle the boundary. This is the exact leak class now guarded by
     `zoo/splits.py` (`assert_no_patient_leak`, `assert_no_series_leak`) — the Phase 6 run predates the guard.
  3. **Razor margin, no interval:** 0.728 cleared 0.720 by 0.008 on one seed; the +0.018 delta
     over the 0.710 baseline was indistinguishable from a lucky seed.
- Retraction semantics that mattered: **not FAIL** (the hypothesis had never been cleanly tested),
  **not KILL** (the arm reopened under a corrected protocol). The number was deleted from the
  claim ledger; the hypothesis went back to *untested*.

## Part 2 — The patient-held-out split (239 patients, 0 leak)

- LoRA-eval patients (malignancy val+test) = **239 patients / 243 series** were held out of
  pretraining **entirely** (23.8% of the corpus). The remaining 775 series split by patient into
  **698 train (694 patients) / 77 val (77 patients)**; 3 series with no DICOM PatientID were
  excluded fail-closed rather than guessed.
- Builder `scripts/preprocessing/build_lidc_patient_holdout_split.py`; manifest
  `data/mvp/split_manifest_lidc_patientholdout.json` (`group_key=patient_id`).
- Verification is a *runnable function*, not a paragraph: `verify_no_patient_leak` in the 3-seed
  runner refuses to start unless manifest holdout patients equal the label-CSV patients and
  pretrain∩holdout = 0. Re-executed fresh on 2026-11-20: **771 pretrain / 239 holdout patients,
  775/243 series, overlap 0.**

## Part 3 — The 30K teacher-entropy dip and the controlled checkpoint sweep

- The 50K-step ViT-Base retrain (4090, 63 min) was **not a clean trajectory**: teacher entropy
  collapsed to **0.87 at step 30K** (loss 1.10, 604 spike warnings) then recovered to 4.70 at 45K
  — the failure family the ViT-Large pan-organ run was killed for. Lesson kept: never default
  checkpoint selection to the terminal step after an unstable run (Phase 6 did).
- So we measured before choosing: layer-9 view retrieval (N=512) on **all ten checkpoints**, with
  the **contaminated Phase 6 50K control re-scored on the identical holdout harness** so the
  comparison is a measurement, not an assumption:

| Checkpoint | ratio vs random | top1 | top5 |
|---|---|---|---|
| 20K (pre-dip peak) | **30.0×** | 0.0586 | 0.1699 |
| 30K (collapse episode) | 31.0× | 0.0605 | 0.1875 |
| 40K (post-recovery peak) | **30.0×** | 0.0586 | 0.1953 |
| 50K (terminal) | 28.0× | 0.0547 | 0.1934 |
| Phase 6 50K control (leaky) | 31.0× | 0.0605 | 0.1875 |

- Reading: removing the leak cost **nothing** on representation quality — the clean backbone
  matches its contaminated ancestor (30.0× vs 31.0×). The best raw point (30K, 31.0×) sits inside
  the collapse episode and was deliberately *not* selected. Pre-registered selection:
  **20K and 40K** (30K excluded by trajectory hygiene; 50K as reference).

## Part 4 — The three-seed verdict: the specialist lost

Gate stated before any LoRA ran (PRD-0001 §2): 3 seeds (42/43/44), slice-level test AUROC on the
121 patient-held-out test patients; GO if mean ≥ 0.720 **and** 95% CI lower bound > 0.710 (the
4-dataset ViT-Small baseline); FAIL if mean < 0.710.

| Checkpoint | seed 42 | seed 43 | seed 44 | mean | 95% CI | gate |
|---|---|---|---|---|---|---|
| 20K | 0.6477 | 0.7055 | 0.7215 | **0.6916** | [0.6476, 0.7355] | FAIL |
| 40K | 0.7129 | 0.6626 | 0.6928 | **0.6894** | [0.6608, 0.7181] | FAIL |
| 50K (ref) | 0.6716 | 0.6595 | 0.6854 | 0.6722 | [0.6575, 0.6868] | FAIL |

Patient-level AUROC (121 patients, honest secondary): 20K mean 0.7223 but CI lower bound 0.6407 —
the apparent bump is patient sampling noise, not signal.

**Verdict: FAIL.** With patients truly held out, single-organ LIDC pretraining does not beat the
4-dataset generalist. The 0.728 was leakage, not skill.

Three sub-findings worth their own sentences:
1. **Seed variance is the story** — slice AUROC at one fixed checkpoint ranges 0.632–0.765
   across seeds. A single-seed protocol could have "confirmed" almost any story. The multi-seed
   rule (red line #5) earned its keep.
2. **The baseline is itself leak-inflated** — the 4-dataset backbone saw LIDC too, and its 0.710
   is a val-set number from the pre-test-split protocol. The clean arm lost to an opponent
   playing with an advantage; the clean-vs-clean gap is at least as bad.
3. **Two of three single-organ specialist expansions have now failed their honest gates**
   (LIDC here, colon at 0.6529). The one release-grade success (hepatic-vessel, external AUROC
   0.9413) is also the one task with genuine external validation — which is exactly the
   distinction the five-gate envelope exists to make.

## Part 5 — What the machinery that survived looks like

- `zoo/splits.py` patient/series guards + fail-closed attribution (a series we can't attribute to
  a patient ends the run, it doesn't get assumed).
- A **controlled control**: the leaky Phase 6 checkpoint re-scored on the clean harness, so
  "did the leak inflate the representation metric?" got measured.
- Pre-registered gates in the PRD *before* GPU spend; verdict vocabulary GO/PIVOT/FAIL; 7-box
  adversarial pass before any verdict is written (bit-exact fresh-process re-eval of two cells;
  aggregates recomputed; leak re-verified).
- Negative results published as credibility assets — the colon FAIL and this LIDC FAIL ship in
  the same ledger style as the wins.

## Closing beat

The most expensive failure in research isn't a wrong model — it's a right-looking number that
was never a measurement. We paid for that lesson twice (pan-organ KILL, LIDC retraction→FAIL)
and the second payment bought the rule that would have caught the first. Next time a specialist
claim gets gated here, it starts from a patient-disjoint split and three seeds, and the brief
you're reading is the evidence that those rules survive contact with the result.

---

## Fact-check table (every number → artifact)

| Claim | Artifact |
|---|---|
| 239 holdout patients / 243 series, overlap 0 | `data/mvp/split_manifest_lidc_patientholdout.json`; `verify_no_patient_leak` (fresh run 2026-11-20) |
| Entropy 0.87 @30K, 4.70 @45K, 604 spike warnings | `runs/20261001_182138_lidc-specialist-vit-base-patient-holdout/` TensorBoard + ledger Phase 10 trajectory table |
| View retrieval 30.0×/30.0× vs control 31.0× | `results/quality_view_retrieval_lidc_holdout.json` |
| 3-seed AUROC table, CIs, FAIL gate | `results/lidc_patient_holdout_lora_3seed.json` (runner `8bcb26a`, job `lidc-holdout-lora-3seed-v1`) |
| Baseline 0.710 (val-set, leak-inflated disclosed) | `adapters/lidc-malignancy-lora-r8-64px-lung-window/finetune_config.json` (best_val auroc 0.7097); ledger Phase 2 ablation table |
| Retraction rationale | `docs/RETRACTION_NOTICE_lidc-specialist-vit-base-scale-aware.md` |
| Verdict + adversarial pass | `docs/EXPERIMENTS_AND_RESULTS.md` Phase 10 (`CONCLUDED — FAIL`) |

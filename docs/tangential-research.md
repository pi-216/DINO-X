# Tangential Research

Cross-pollination from external research and internal experimental findings relevant to DINO-X's architecture, training methodology, and evaluation pipelines.

---

## Monte Carlo Dropout Uncertainty and Entropy-Thresholded Selective Prediction for Architecture-Agnostic Brain Tumor MRI Triage (2026)

**Source**: https://arxiv.org/abs/2607.16317  
**Author**: Medhansh Sharma

### What It Is

A study investigating predictive uncertainty and "defer-to-human" selective prediction for 4-class brain tumor MRI triage (glioma, meningioma, pituitary, no tumor). The author evaluated ViT-B/16 and ResNet-50 backbones under a strict leakage-controlled partition.

The core premise is that for clinical deployment, raw accuracy is less important than a model's ability to know *when it doesn't know* and safely defer ambiguous cases to a radiologist. 

### Core Findings

1. **Deterministic Confidence Heads Collapse**: An auxiliary confidence head (trained alongside the classifier to predict correctness) collapsed to a near-constant output, failing to track epistemic uncertainty. Minimizing the task loss was easier than predicting true confidence.
2. **Data Leakage in Medical MRI is Severe**: 48.9% of the Kaggle dataset used was near-duplicates (augmented crops/slices). A naive file-level random split heavily inflated test accuracy. The author solved this using perceptual-hash (pHash) clustering (Hamming distance ≤ 5) to ensure duplicates stayed in the same split.
3. **MC Dropout Uncertainty is Actionable**: Using Monte Carlo Dropout (running the image through the network 20 times with dropout on at inference) and calculating predictive entropy yielded excellent uncertainty bounds. 
4. **Defer-to-Human Triage**: By simply deferring the most uncertain 5% of cases (based on MC Dropout entropy) to a human reader, accuracy on the remaining 95% of cases jumped from ~96.2% to ~98.0%.
5. **Temperature Scaling Fixes Calibration**: Fitting a single temperature scalar ($T \approx 0.62$) to the deterministic logits on the validation split dropped the Expected Calibration Error (ECE) from ~0.07 to ~0.018 without changing a single prediction.
6. **Architecture is Secondary to Pipeline**: A rigorous per-seed McNemar test found ViT-B/16 and ResNet-50 statistically indistinguishable. The clinical value came from the uncertainty pipeline, not the specific network.

### Relevance to DINO-X

| Finding | Actionability |
|---------|--------------|
| **pHash Leakage Control** | **Critical & Direct** — CT scans in DINO-X (LIDC-IDRI/MVP) will have nearly identical adjacent slices and crops. Random splits will hallucinate success. We must group by patient ID or perceptual hash. |
| **MC Dropout + Entropy** | **Cheap Drop-in** — ViT-Large naturally supports dropout. In `evaluate_panorgan.py`, we can run 20 passes and use entropy as a clinical gating mechanism ("defer to human"). |
| **Avoid Aux Confidence Heads** | **Architectural Guardrail** — Confirms we shouldn't waste time engineering deterministic confidence heads for nodule ambiguity. |
| **Temperature Scaling** | **Quick Win** — Easily calibrates the raw `panorgan` output probabilities before hitting clinical evaluation layers. |
| **Attention Rollout for Debugging** | **Validation Tool** — Dumping ViT attention maps on high-entropy errors can visually confirm if DINO-X is failing due to looking at lung walls instead of the nodule. |

### Concrete Ideas for DINO-X

1. **Verify Patient-Level/pHash Splits**: Audit the `mvp_combine_indices.py` or equivalent data preparation scripts to guarantee no multi-slice/augmented nodule leakage occurs between train/val/test splits.
2. **Implement MC Dropout in Evaluation**: Modify `scripts/evaluate_panorgan.py` to support a `--mc-dropout` flag that runs inference $T=20$ times, calculating predictive entropy to plot Risk-Coverage curves.
3. **Fit Temperature Scalar**: Add a simple post-hoc temperature scaling step to the validation loop in `scripts/finetune_lora.py` or evaluation scripts to output perfectly calibrated confidence scores.
4. **Attention Rollout on Errors**: For `[CONCLUDED — FAIL]` models or tricky datasets, export the ViT attention maps on the most "uncertain" predictions to visually debug the failure modes.

---

## Zero-Shot DINOv3-Based Image Matching via Many-to-Many Association (2026)

**Source**: https://arxiv.org/abs/2604.23670  
**Authors**: Haodong Jiang, Mingzhe Li, Junfeng Wu

### What It Is

A paper exploring zero-shot geometric image matching using frozen DINOv3 features. The authors demonstrate that while DINO features are incredibly rich semantically, they are geometrically ambiguous out-of-the-box (e.g., the left and right sides of a symmetrical object map to the same semantic feature). By abandoning 1-to-1 matching and mitigating DINO's positional artifacts, they achieve robust geometric matching performance, especially on out-of-distribution (OOD) datasets.

### Core Findings

1. **Final Layers Destroy Spatial Correspondence**: Geometric correspondence quality drops sharply in the final layers of the ViT. For the ViT-L/16 model (24 layers), they identified the 19th layer as the absolute best for matching. Using the final layer severely degrades matching accuracy.
2. **DINO Has a Destructive Positional Bias**: DINO patch features contain a stable positional artifact that interferes with geometric matching. Projecting the features onto the null space of this bias (found via PCA on a noise image) recovers massive amounts of geometric accuracy.
3. **The 1-to-1 Matching Fallacy**: Because DINO is highly semantic, enforcing a strict 1-to-1 nearest neighbor test frequently discards the true geometric match. Relaxing this to admit multiple candidate associations (Many-to-Many / top-$K$ mutual nearest neighbors) substantially improves recall.

### Relevance to DINO-X

| Finding | Actionability |
|---------|--------------|
| **Layer Selection for Retrieval** | **Critical & Direct** — DINO-X's view retrieval evaluation recently peaked at 31× and failed the 40× gate. If `phase5_view_retrieval_eval.py` uses the final ViT layer (layer 12 for ViT-Base), it is severely handicapping spatial matching capability. |
| **Positional Bias Correction** | **High ROI** — CT nodule crops carry strong positional priors. Positional artifacts in DINO might be conflating spatial position with true semantic feature similarity during retrieval. |
| **Many-to-Many Association** | **Medium** — If the retrieval script relies on strict 1-to-1 patch correspondence to calculate similarity, it might be penalizing geometrically ambiguous but semantically matching nodule patches. |

### Concrete Ideas for DINO-X

1. **Change Retrieval Evaluation Layer**: Immediately modify `scripts/phase5_view_retrieval_eval.py` to extract features from a middle-back layer (e.g., layer 9 or 10 for ViT-Base, instead of layer 12). This quick modification might single-handedly push the view retrieval metric from 31× past the 40× gate requirement.
2. **Implement Positional Bias Projection**: Test evaluating view retrieval after projecting out the positional bias as an evaluation-time preprocessing step.
3. **Relax Retrieval Matching**: If view retrieval relies on spatial matching scoring, allow Top-$K$ (e.g., $K=5$) patch assignments before computing the final view distance.

---

## Xray-Visual Models: Scaling Vision Models on Industry Scale Data (2026)

**Source**: https://arxiv.org/abs/2602.16918  
**Authors**: Shlok Mishra, Tsung-Yu Lin, Linda Wang, et al. (Meta AI)

### What It Is

Meta's systematic study of training vision models on industry-scale social media data — starting from 100B+ raw images/videos, curated down to 15B image-text pairs (ViSE) + 10B video-hashtag pairs (URU). Three-stage training: MAE → hashtag classification → CLIP contrastive. EViT-2B backbone achieving SOTA with 4× fewer tokens than competitors.

### Core Findings

1. **Semantic Resampling Beats Raw Scale**: WordNet synset-based resampling (undersample head concepts, oversample tail) gave **+7.3% ImageNet accuracy** on ViT-B from scratch. Data balance matters more than raw volume.

2. **Batch Mixing Is a First-Class Hyperparameter**: Strict alternation between modalities cost 4.5% Kinetics accuracy vs probabilistic sampling that exhausts both simultaneously per epoch. Prolonged single-modality exposure creates bias.

3. **Noise Filtering > More Data**: MetaCLIP similarity filtering at 0.25 threshold + label smoothing gave +1.1% over unfiltered. Quality gates beat quantity at the margin.

4. **Multi-Stage Data Routing**: Different data at different stages — noun-heavy hashtags for MAE/classification, action-heavy captions for CLIP. Task-aligned data routing, not uniform mixing.

5. **Synthetic Data Requires Refinement**: MMLLM-generated captions improved retrieval but hurt classification due to 3% repetition. LLM-based rewriting (Llama 3B) fixed this. Siamese loss on synthetic + original captions recovered the trade-off.

6. **Academic Benchmarks ≠ Real-World Performance**: Xray-Visual's key finding — SOTA on ImageNet/Kinetics did NOT predict performance on internal retrieval tasks. Models trained on diverse real-world data outperformed academically-SOTA models by +10% on cross-domain tasks.

### Relevance to DINO-X

| Finding | Actionability |
|---------|--------------|
| **Semantic resampling** | **Validates existing approach** — DINO-X's T=2.0 temperature-scaled sampling is the same principle. Xray-Visual proves it works at 26B scale. |
| **Batch mixing strategy** | **High** — DINO-X's `diverse_batches=True` is exactly the probabilistic sampling approach. Document this as a model zoo convention. |
| **Multi-stage data routing** | **Missed opportunity** — DINO-X uses the same data at all stages. LIDC specialist could use noun-heavy crops early, then shift to clinically annotated crops. |
| **Academic ≠ real-world** | **Core validation** — DINO-X's scanner fingerprinting discovery is exactly this: view retrieval looked fine while clinical utility collapsed. Internal metrics (LoRA AUROC) are the real gate. |
| **Synthetic data refinement** | **Future direction** — If DINO-X ever uses synthetic CT augmentation (diffusion-generated nodules), expect quality issues that need LLM-style refinement. |

### Concrete Ideas for DINO-X

1. **Document batch mixing as a model zoo convention**: Every specialist run should log its batch sampling strategy. The probabilistic-exhaustion approach (not strict alternation) is the proven default.
2. **Evaluate progressive resolution**: Xray-Visual's 98→154→224→336→448 progressive schedule could address DINO-X's crop-size sensitivity (64px >> 128px >> 224px for nodule classification). Start at 64px, scale up during training.
3. **Data quality filters before scale**: Before adding any new dataset to the model zoo, implement a SOLF-like geometric sanity filter (see UKBOB entry below). Quality gates before quantity.

---

## UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation (2025)

**Source**: https://arxiv.org/abs/2504.06908  
**Authors**: Emmanuelle Bourigault, Amir Jamaludin, Abdullah Hamdi (Oxford VGG)

### What It Is

The largest labeled medical imaging segmentation dataset: 51,761 MRI 3D samples from UK Biobank, 17.9M 2D images, 1.37B segmentation masks across 72 organs. Uses automatic labeling (TotalVibeSegmentator) + statistical filtering (SOLF) + entropy test-time adaptation (ETTA) to handle label noise at scale. Trains Swin-BOB foundation model achieving SOTA on BRATS and BTCV.

### Core Findings

1. **SOLF: Geometric Label Filtering**: Specialized Organ Label Filter uses normalized volume, sphericity, and eccentricity to flag erroneous organ labels. Filtered labels achieve Dice 0.891 vs unfiltered 0.873 on manual validation. Filters catch label errors, not patient abnormalities.

2. **ETTA: Entropy Test-Time Adaptation**: Fine-tune only BatchNorm parameters at test time using entropy minimization on test data. +1.3% Dice on BTCV. Handles domain shift without retraining.

3. **Automatic Labels + Filtering > Manual Labels at Small Scale**: 51K automatically labeled MRIs with SOLF filtering outperforms models trained on small manually labeled datasets. Scale + quality control beats precision at small scale.

4. **Zero-Shot Generalization Validates Labels**: Models trained on filtered UKBOB generalize zero-shot to AMOS (MRI) and BTCV (CT) — proving the automatic labels capture real anatomy, not artifacts.

5. **Filtration Threshold Matters**: ϵ=2 (2nd percentile) is optimal. Too tight (ϵ=5) filters out valid but unusual anatomy; too loose (ϵ=0) keeps noisy labels. Dice drops from 0.892 → 0.745 across this range.

### Relevance to DINO-X

| Finding | Actionability |
|---------|--------------|
| **SOLF geometric filtering** | **High** — DINO-X has no label quality control for its automatic preprocessing. A geometric sanity filter (slice spacing consistency, HU distribution bounds, anatomical plausibility) could catch corrupted slices before they poison training. |
| **ETTA for scanner shift** | **High** — The exact problem that killed pan-organ: LIDC↔colon↔pancreas↔CQ500↔MSD all have different scanners. ETTA's BatchNorm-only adaptation is a cheap post-hoc fix for DINO-X specialists deployed on new hospitals. |
| **Automatic labels at scale** | **Validates DINO-X's approach** — DINO-X uses automatic preprocessing (HU windowing, cropping, slice extraction). UKBOB proves this scales if you add quality filters. |
| **Zero-shot generalization as validation** | **Methodological** — DINO-X's view retrieval eval is a form of this. UKBOB's approach (train on filtered, test on external) is a cleaner validation protocol. |
| **Foundation model path** | **Strategic** — UKBOB proves pan-organ CAN work with enough data diversity + quality control + adaptation. DINO-X's failure was specific to DINO objective + 5-dataset imbalance, not fundamental. |

### Concrete Ideas for DINO-X

1. **Implement SOLF-like filters for CT slices**: Add geometric sanity checks to `zoo/data.py` — slice spacing consistency (flag slices with abnormal z-gaps), HU distribution bounds (flag slices with implausible HU ranges), anatomical plausibility (flag slices that are mostly air or mostly bone when they shouldn't be).

2. **Add ETTA to specialist deployment**: For any DINO-X specialist deployed on a new hospital's data, create an `adapt_etta.py` script that fine-tunes BatchNorm parameters using entropy minimization on the target domain. Ship this with every model in the zoo.

3. **UK Biobank as future MRI data source**: 51K neck-to-knee MRI scans with 72 organ labels. If DINO-X expands to MRI (roadmap mentions cross-modality), this is the dataset. The 300-sample manual validation set (UKBOB-manual) is a gold standard for evaluating any MRI specialist.

4. **Document data provenance per model**: UKBOB's paper structure (dataset → filtering → foundation model → benchmarks) is a clean template for documenting each model in the DINO-X zoo. Every specialist should have: data provenance, quality filters applied, training config, benchmark results, known failure modes.

5. **Use zero-shot generalization as a gate**: Before adding a new specialist to the model zoo, require zero-shot transfer to at least one external dataset from a similar domain (e.g., LIDC specialist → generalize to a different lung CT dataset).

---

## RAU: Reference-based Anatomical Understanding with Vision Language Models (2026)

**Source**: https://arxiv.org/abs/2509.22404 (ECCV 2026)
**Authors**: Yiwei Li, Yikang Liu, Jiaqi Guo, et al. (United Imaging Intelligence + UGA/Northwestern)

> **Status: PARKED — tangential to the current release effort.** Logged for reference only. Do not pursue until at least one DINO-X specialist ships; revisit at the cross-modality expansion decision.

### What It Is

A reference-based paradigm for anatomical understanding: instead of training a model to "know" every structure a priori, give it a **labeled reference image** and make it find the corresponding structure in an **unlabeled target** via *relative spatial reasoning* (how a human annotator works). Built on Qwen2.5-VL-7B (LoRA r=8/α=16, SFT → GRPO RL) in three escalating modes:

1. **VQA** — label a target region given a reference (RAOS-CT-only training → 16.6% → 70.7% labeling acc ID, 63.9% avg OOD).
2. **BBox + optimal-transport global matching** — raises ID accuracy but **regresses on vessels** (Arcade bbox only 41% vs 82% for the segmentation variant). Their own negative result: boxes are wrong for thin/fragmented structures.
3. **VLM ⊗ SAM2** (the contribution) — VLM emits `<Seg>` tokens → MLP → soft-prompt injected into SAM2's memory bank → dense mask. Trained on RAOS-CT (~450k whole-body CT slices, 19 organs) + Arcade-X-Ray (~21k coronary-vessel images). Avg Dice 0.31 → **0.71**, gIoU 0.21 → 0.50 vs the MedSAM2-Memory-Ref-SFT baseline, across ID (RAOS, Arcade) + OOD (CAMUS ultrasound, LERA X-ray). ~10× annotation-time reduction.

### Core Findings

1. **Reference-based spatial reasoning generalizes across modalities** — CT→ultrasound→X-ray OOD transfer from relative-location cues alone, without per-dataset re-training; RL (GRPO) end-task rewards beat SFT (which overfits/memorizes training images).
2. **Bounding boxes are wrong for vessels** — boxes can't capture thin/curved/fragmented anatomy; pixel-level reference-conditioned segmentation is the fix.
3. **Reference quality dominates** — LPIPS-stratified: accuracy holds ~90%/+85% for well-matched reference↔target but drops to ~56–61% for mismatched pairs; acknowledged failure cases (Fig S6).
4. **Cheap labels: 10× annotation-time reduction** — RAU masks as initialization + SAM2 interactive refinement; a per-target data-procurement layer.
5. **Reference retrieval = frozen DINO features + FAISS cosine** — the front end is DINO-family features, which DINO-X already produces.

### Relevance to DINO-X

| Finding | Actionability |
|---------|--------------|
| **Low-texture / small-N remedy** | **Direct fit for the colon FAIL** — colon (126 patients, AUROC 0.653 < 0.70 gate) is exactly the low-texture, label-scarce anatomy RAU targets. Valid alternative hypothesis to "colon is hopeless," but NOT a release blocker — defer. |
| **Segmentation framing for vessels** | **Signal for hepatic-vessel** — RAU shows box/crop-classification is a weak framing for vasculature; reference-conditioned pixel segmentation is the better gate. Revisit only if hepvessel LoRA AUROC proves insufficient. |
| **Cheap-label pipeline for the zoo** | **Useful, not now** — RAU as a pure annotation pre-labeler could feed cleaner labels into supervised specialists. |
| **Cross-modality path** | **Deferrable** — CT→US→X-ray OOD via reference is a lighter path than training separate modality models, directly relevant to the MRI/X-ray roadmap. |
| **Feature-level adoption without a 7B VLM** | **Highest long-term leverage** — DINO-X already has DINO/ScaleEmbedding encoders + view retrieval, so reference-conditioning could be adopted at the feature level inside the current ViT stack, skipping the Qwen/SAM2 weight. |

### Concrete Ideas for DINO-X (deferred)

1. **Alternative colon arm (pre-registered gate ≥0.70 AUROC)** — BEFORE writing colon off, consider a feature-level reference-conditioned variant reusing DINO-X's own encoders as the reference-retrieval front end. Gate it like any other arm.
2. **Add a segmentation companion gate for hepatic-vessel** — if crop-classification AUROC proves sufficient it's moot; if not, reference-conditioned segmentation is the documented fallback.
3. **Reuse as a cheap-label pipeline** — when (if) RAU releases weights/masks, reproduce under external held-out audit (adversarial pass) before trusting the 0.71 Dice; numbers are a claim, no code/masks released as of Sep 2026.
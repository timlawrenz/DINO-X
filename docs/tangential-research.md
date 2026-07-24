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
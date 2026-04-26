# Experiments Log

Updated: 2026-04-26

---

## 5-Dataset Phase 3 Extended: Doubled Batch Size (2026-04-26)

**Goal:** Test whether doubling the effective batch size from 128 to 256 (the DINO
paper's recommended minimum) improves representations. This was the "next experiment"
identified in the LoRA benchmark analysis.

### Training Configuration

- **Model:** ViT-Small (dim=384, depth=12, heads=6, patch=14, ~70.2M with projector)
- **Effective batch:** 256 (batch=64 × accum=4) — **doubled from Phase 3's 128**
- **Optimizer:** AdamW, lr=2e-4, cosine decay to 1e-6, warmup=2,500 steps
- **Loss:** DINO + Gram(1.0) + KoLeo(0.1)
- **Anti-memorization:** crop_scale_min=0.3, z_stride=3, diverse_batches=True
- **AMP:** bfloat16
- **Steps:** 50,000 (resumed from step 40,338)
- **Hardware:** AMD Radeon 8060S (Strix Halo), ROCm 7.11
- **Data:** 5-dataset temperature-scaled index (400K slices, same as Phase 3)
- **Run directory:** `runs/20260423_171906_5dataset-phase3-small-bs256/`
- **Git commit:** `e8fc8d1`

### View Retrieval (Step 50K, N=512)

| Metric | Value |
|--------|-------|
| top-1 accuracy | 10.5% |
| top-5 accuracy | 39.5% |
| Random baseline | 0.2% |
| **Ratio vs random** | **54.0×** |
| Passed | ✅ |

### Comparison: Batch 128 vs Batch 256

| Metric | Eff Batch=128 | Eff Batch=256 |
|--------|--------------|---------------|
| View retrieval (50K) | 56.0× | 54.0× |
| Run directory | `20260422_202622_...` | `20260423_171906_...` |

The doubled batch size produces comparable view retrieval (54× vs 56×). The slight
decrease is within noise for N=512 evaluation.

### LoRA Fine-Tuning (LIDC Malignancy, bs256 Backbone)

Same protocol as the original benchmark: LoRA rank=8, alpha=16, 64px nodule crops,
lung HU window (level=-30, width=120 scaled), 50 epochs, early stopping on val AUROC
(patience=10). 1,300 train / 262 val nodules.

| LR | Val AUROC | Val Accuracy | Best Epoch |
|----|-----------|-------------|------------|
| 5e-4 | 0.684 | 0.653 | 14 |
| 1e-3 | 0.674 | 0.607 | 43 |

### Full Comparison: All Backbones

| Backbone | Eff Batch | View Retrieval | Best LoRA AUROC | Best LR |
|----------|-----------|---------------|-----------------|---------|
| **4-dataset** | **128** | **7.0×** | **0.710** | **5e-4** |
| 5-dataset (batch=128) | 128 | 56.0× | 0.680 | 1e-3 |
| 5-dataset (batch=256) | 256 | 54.0× | 0.684 | 5e-4 |

### Analysis

1. **Doubling batch size did not close the gap:** The bs256 backbone (0.684) is
   essentially identical to the bs128 backbone's best (0.680), both ~3% below the
   4-dataset specialist (0.710). The DINO paper's batch size recommendation does
   not help with the capacity dilution problem.

2. **Capacity dilution is the bottleneck:** ViT-Small (22M params) cannot maintain
   organ-specific features when trained across 5 organs. The path forward is
   ViT-Large (or larger), not hyperparameter tuning of ViT-Small.

3. **View retrieval ≠ downstream performance:** The 5-dataset models have dramatically
   better general representations (54-56× vs 7×) but worse lung-specific LoRA
   performance. View retrieval measures representation generality, not task fitness.

### Artifacts

- `runs/20260423_171906_5dataset-phase3-small-bs256/checkpoint_final_00050000.pth`
- `runs/20260423_171906_5dataset-phase3-small-bs256/view_retrieval_step50000_N512.json`
- `adapters/lidc-malignancy-5dataset-bs256-step50000-lr5e-4/` — AUROC 0.684
- `adapters/lidc-malignancy-5dataset-bs256-step50000-lr1e-3/` — AUROC 0.674

---

## LoRA Benchmark: 5-Dataset Backbone vs 4-Dataset (2026-04-23)

**Goal:** Determine whether adding CQ500 brain CTs (5-dataset) improves or degrades
downstream task performance vs the 4-dataset lung-focused backbone, measured by LoRA
fine-tuning on LIDC-IDRI nodule malignancy classification.

### Backbones Under Test

| Backbone | Datasets | Steps | Eff Batch | View Retrieval (final) | Hardware |
|----------|----------|-------|-----------|----------------------|----------|
| 4-dataset ablation | LIDC, Pancreas, MSD-Hepatic, MSD-Colon | 5K | 128 | 7.0× | RTX 2070 SUPER |
| 5-dataset phase3 | + CQ500 (brain CT) | 50K | 128 | 56.0× | Radeon 8060S (Strix Halo) |

5-dataset training: `runs/20260422_202622_5dataset-phase3-small` on Strix Halo.
400K slices, ViT-Small, bfloat16, KoLeo=0.1, crop_min=0.3, z_stride=3.

### View Retrieval Trajectory (5-Dataset)

| Step | Ratio | Note |
|------|-------|------|
| 7,500 | 64.0× | Post-warmup peak |
| 10,000 | 63.0× | Stable |
| 25,000 | 52.0× | Mid-training dip (cosine decay) |
| 50,000 | 56.0× | Recovery, final checkpoint |

Note: Steps 2,500 and 5,000 showed ratio=2.0× due to a bug in the eval script
(spacing not passed to backbone). Fixed in commit `b6ce59d`.

### LoRA Fine-Tuning Configuration

- **Task:** LIDC-IDRI nodule malignancy (binary: benign vs malignant, threshold=3)
- **Adapter:** LoRA rank=8, alpha=16, targets=qkv+proj+fc1+fc2
- **Data:** 1,300 train / 262 val / 315 test nodules (64px crops, lung HU window)
- **Window:** level=-30, width=120 (scaled HU; real: level=-300, width=1200)
- **Training:** 50 epochs, early stopping on val AUROC (patience=10)
- **Frozen:** ScaleEmbedding, PatchEmbed, positional embeddings, registers

### Results: Checkpoint Sweep (LR=5e-4)

| Backbone | Step | Val AUROC | Val Accuracy | Best Epoch |
|----------|------|-----------|-------------|------------|
| **4-dataset (Model A)** | **5K** | **0.710** | **0.660** | **24** |
| 5-dataset | 10K | 0.672 | 0.595 | 23 |
| 5-dataset | 25K | 0.639 | 0.584 | 10 |
| 5-dataset | 50K | 0.635 | 0.580 | 10 |

### Results: LR Sweep (5-Dataset, Step 50K)

| LR | Val AUROC | Best Epoch |
|----|-----------|------------|
| 1e-3 | 0.680 | 18 |
| 5e-4 | 0.635 | 10 |
| 2e-4 | 0.668 | 7 |
| 1e-4 | 0.619 | 18 |

### Analysis

1. **Capacity dilution confirmed:** The 5-dataset backbone has dramatically better
   general-purpose representations (56× view retrieval vs 7×) but worse lung-specific
   downstream performance (best 0.680 vs 0.710 AUROC). Adding brain CTs forced the
   ViT-Small to spread its 22M parameters across more domains.

2. **More training steps hurt lung-specific features:** Within the 5-dataset run,
   the 10K checkpoint (0.672) outperformed 25K (0.639) and 50K (0.635) at the default
   LR. The model continued improving its general representations while degrading
   organ-specific features.

3. **LR matters but doesn't close the gap:** The best LR for the 50K checkpoint
   (1e-3 → 0.680) is higher than the 4-dataset optimal (5e-4), suggesting the
   5-dataset features require more aggressive adaptation. But even the best LR
   leaves a 3% gap.

4. **Effective batch size was 128, not 256:** Original DINO paper recommends ≥256.
   With batch=32 and accum=4, teacher centering and KoLeo only see 32 samples per
   micro-batch. This may limit multi-domain learning. A batch-size-doubled run
   (eff=256) is the next experiment.

### Artifacts

- `adapters/lidc-malignancy-lora-r8-64px-lung-window/` — Model A (4-dataset, AUROC 0.710)
- `adapters/lidc-malignancy-5dataset-step10000/` — 5-dataset step 10K (AUROC 0.672)
- `adapters/lidc-malignancy-5dataset-step25000/` — 5-dataset step 25K (AUROC 0.639)
- `adapters/lidc-malignancy-5dataset-step50000/` — 5-dataset step 50K (AUROC 0.635)
- `adapters/lidc-malignancy-5dataset-50k-lr1e-3/` — 5-dataset 50K, LR=1e-3 (AUROC 0.680)
- `adapters/lidc-malignancy-5dataset-50k-lr1e-4/` — 5-dataset 50K, LR=1e-4 (AUROC 0.619)
- `adapters/lidc-malignancy-5dataset-50k-lr2e-4/` — 5-dataset 50K, LR=2e-4 (AUROC 0.668)
- `runs/20260422_202622_5dataset-phase3-small/` — Training run + view retrieval JSONs

---

## 5-Dataset Phase 3 Pretraining (2026-04-22)

**Goal:** Train ViT-Small on 5 datasets (400K slices across 4 organs) with all
anti-memorization constraints active. First pan-organ training run.

### Datasets (Temperature-Scaled, T=2.0)

| Dataset | Organ | Raw Slices | T=2.0 Sampled | Factor |
|---------|-------|-----------|---------------|--------|
| LIDC-IDRI | Lung | 243,990 | 163,220 | 0.67× ↓ |
| CQ500 | Brain | 103,711 | 106,415 | 1.03× |
| MSD-Hepatic | Liver | 21,120 | 48,021 | 2.27× ↑ |
| Pancreas-CT | Abdomen | 17,764 | 44,041 | 2.48× ↑ |
| MSD-Colon | Colon | 13,486 | 38,373 | 2.85× ↑ |
| **Total** | | **400,071** | **400,070** | |

### Training Configuration

- **Model:** ViT-Small (dim=384, depth=12, heads=6, patch=14, ~70.2M with projector)
- **Effective batch:** 128 (batch=32 × accum=4)
- **Optimizer:** AdamW, lr=2e-4, cosine decay to 1e-6, warmup=2,500 steps
- **Loss:** DINO + Gram(1.0) + KoLeo(0.1)
- **Anti-memorization:** crop_scale_min=0.3, z_stride=3, diverse_batches=True
- **AMP:** bfloat16 (no grad scaler)
- **Steps:** 50,000 (~53 epochs)
- **Hardware:** AMD Radeon 8060S (Strix Halo), ROCm 7.11
- **Throughput:** ~390 samples/s (stabilized), total ~4.8 hours
- **Git commit:** `20ee253a`
- **Data manifest hash:** `817d2641b44806db`

### Training Dynamics

| Step | Loss | LR | Samples/s |
|------|------|----|-----------|
| 0 | 8.78 | 8.0e-8 | 3.1 |
| 1,000 | 6.81 | 8.0e-5 | 369 |
| 2,000 | 3.32 | 1.6e-4 | 394 |
| 5,000 | ~2.0 | 2.0e-4 | ~390 |

Monitor at step 1,000: Embed-L0 std=0.099, Gram mean=0.996 (healthy).
Monitor at step 2,000: Embed-L0 std=0.052, Gram mean=0.999 (attention tightening).

No NaN events. `F.log_softmax` fix held throughout bfloat16 training.

### Artifacts

- `runs/20260422_202622_5dataset-phase3-small/` — Full run directory
- `data/mvp/combined_5dataset_t2.csv` — Temperature-scaled index
- `data/mvp/split_manifest_5dataset.json` — Train/val split

---

## 4-Dataset Anti-Memorization Ablation (2026-04-22)

**Goal:** Fix catastrophic collapse at epoch 3 in multi-organ training by blocking
shortcut learning pathways. The previous 4-dataset run (without anti-memorization)
collapsed to loss=0.05 and retrieval ratio=1.0× by epoch 3.

### Anti-Memorization Patches Applied

1. **KoLeo regularization** (weight=0.1): Forces embeddings to spread uniformly on
   the hypersphere, preventing trivial domain clustering.
2. **Aggressive crop** (scale_min=0.3): Wider RandomResizedCrop forces network to
   match extreme local details with global context.
3. **Z-stride=3**: Skip 2 slices between consecutive samples to break Z-axis
   contiguous memory leak.
4. **Diverse batches**: Ensure each mini-batch contains slices from multiple datasets.

### Results

| Step | Loss | View Retrieval Ratio |
|------|------|---------------------|
| 1,000 | 0.58 | — |
| 2,000 | 0.47 | — |
| 3,000 | 0.32 | 5.0× |
| 5,000 | 0.18 | **7.0×** ✓ |

**Success:** Ratio climbing at 7.0× with no collapse. Loss at 0.18 is low but the
climbing retrieval ratio proves the model is learning structural features, not
memorizing. The architecture is cleared for Phase 3 (5-dataset training).

### Artifacts

- `runs/vit-small-4dataset-ablation/20260422_104149/` — Run directory
- `runs/vit-small-4dataset-ablation/20260422_104149/checkpoint_final_00005000.pth` — Final checkpoint (Model A backbone)

---

## LIDC-IDRI Nodule Malignancy LoRA Benchmark (2026-04-22)

**Goal:** First proof-of-value for frozen DINO-X backbone. Fine-tune a LoRA adapter
on LIDC-IDRI nodule malignancy classification to beat the linear probe baseline
(AUROC 0.687) and approach the supervised ResNet18 literature baseline (0.767).

### Data Pipeline

- **Source:** pylidc annotations, consensus ≥3 raters, malignancy threshold=3
- **Crops:** Nodule-centered bounding box with 2× padding, resized to target size
- **Split:** Patient-stratified 70/13/17 — 1,300 train / 262 val / 315 test

### Crop Size Ablation (4-dataset backbone, LR=5e-4 unless noted)

| Crop Size | HU Window | LR | Val AUROC | Notes |
|-----------|-----------|-----|-----------|-------|
| 64px | Lung (L=-30, W=120) | 5e-4 | **0.710** | **Best** — nodule fills FOV |
| 64px | Wide (L=40, W=400) | 5e-4 | 0.684 | Generic window compresses signal |
| 64px | Wide (L=40, W=400) | 1e-3 | 0.681 | First attempt |
| 128px | Wide (L=40, W=400) | 1e-3 | 0.566 | Nodule still too small |
| 224px | Wide (L=40, W=400) | 1e-3 | 0.590 | Nodule tiny; crop can exclude it |

### Key Findings

1. **Crop size dominates:** 64px crops (nodule fills FOV) >> 128px >> 224px.
   Upscaling blur is less harmful than having the nodule be a tiny fraction of input.
2. **Task-specific HU windowing adds +3%:** Lung window vs generic wide window.
3. **3-slice vs 1-slice gap:** Backbone pretrained on 3 consecutive z-slices (2.5D),
   but fine-tuning loads single slice replicated 3×. Likely explains nodule-level gap.

### Held-Out Test Set Results (Best Model: 64px, Lung Window)

| Level | AUROC |
|-------|-------|
| Nodule-level | 0.667 |
| Patient-level | **0.706** |

Patient-level AUROC 0.706 beats the linear probe baseline (0.687). Gap to ResNet18
literature baseline (0.767) suggests room for improvement with 2.5D input matching
and larger backbone.

### Artifacts

- `adapters/lidc-malignancy-lora-r8-64px-lung-window/` — Best adapter
- `scripts/preprocessing/extract_lidc_malignancy.py` — Extraction pipeline
- `data/lidc-idri/labels/malignancy_{train,val,test}.csv` — Split CSVs

---

## MVP Two-Organ Scale-Aware Ablation (2026-04-21)

**Goal:** Prove the full model zoo pipeline end-to-end: two real CT datasets (different
organs, different scales), proper lineage tracking, and scale-aware training comparison.

### Datasets

| Dataset | Organ | Series | Slices | Pixel Spacing | Slice Thickness |
|---------|-------|--------|--------|---------------|-----------------|
| LIDC-IDRI | Lung | 100 | 24,441 | 0.46–0.98mm | 0.625–5.0mm |
| Pancreas-CT | Abdomen | 80 | 18,942 | 0.6–0.98mm | ~1.0–3.0mm |
| **Combined** | **Multi-organ** | **180** | **43,383** | **0.46–0.98mm** | **0.625–5.0mm** |

Data hosted on HuggingFace: `timlawrenz/dinox-mvp-data` (10.4GB processed PNG).

### Training Configuration

- **Model:** ViT-Small (dim=384, depth=12, heads=6, patch=14, params=70.2M with projector)
- **Effective batch:** 256 (batch=64 × accum=4)
- **Optimizer:** AdamW, lr=2e-4, cosine decay to 1e-6, warmup=500 steps
- **Loss:** DINO + Gram(1.0) + KoLeo(0.1)
- **EMA:** 0.996, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
- **Steps:** 5,000
- **AMP:** fp16
- **Seed:** 42
- **Git commit:** `e50f3b278af7327e28f88a94b7b335dd69828304`
- **Data manifest hash:** `f6e833d0add3a664`

### Results

| Arm | Hardware | Final Loss | Steps/s | Time | View Retrieval Ratio |
|-----|----------|------------|---------|------|---------------------|
| **Baseline** (no scale embed) | RTX 3090 Ti | **8.992** | 2.49 | 33.4 min | **5.0×** random |
| **Scale-Aware** (with ScaleEmbedding) | RTX 5070 | **0.134** | 1.33 | 62.8 min | **4.0×** random |

#### View Retrieval (N=2048, seed=42, validation split)

| Metric | Baseline | Scale-Aware |
|--------|----------|-------------|
| Top-1 accuracy | 0.244% | 0.195% |
| Top-5 accuracy | 0.977% | 0.732% |
| Random baseline | 0.049% | 0.049% |
| Ratio vs random | 5.0× | 4.0× |
| Pass threshold (10×) | ❌ | ❌ |

#### Health Monitors (final checkpoint)

| Monitor | Baseline | Scale-Aware | Healthy Range |
|---------|----------|-------------|---------------|
| Embed-L0 Diversity (std) | 0.034 | 0.065 | >0 |
| Output Gram (mean) | 0.9996 | 0.9950 | <1.0 |

### Analysis

1. **Loss convergence:** The scale-aware model achieves dramatically lower DINO loss
   (0.134 vs 8.992). The baseline loss plateaus near the entropy wall (ln(8192)=9.01),
   while the scale-aware model breaks through it decisively. The ScaleEmbedding provides
   a strong additional signal that helps the student match the teacher.

2. **View retrieval parity:** Despite the massive loss difference, view retrieval ratios
   are comparable (5.0× vs 4.0×). This is expected — the view retrieval eval creates two
   augmented crops of the same image (same spacing), so the scale embedding adds identical
   information to both views and doesn't directly help distinguish them. The metric
   primarily tests visual feature quality in isolation from spacing.

3. **No collapse:** Both models are healthy. Embed-L0 diversity is positive (patch
   embeddings are diverse), and output Gram mean is <1 (attention hasn't collapsed).
   The scale-aware model actually shows higher embedding diversity (0.065 vs 0.034),
   suggesting richer learned representations.

4. **Cost:** Baseline ~$0.04 (RTX 3090 Ti @ $0.076/hr × 33 min). Scale-aware ~$0.01
   (RTX 5070 @ $0.012/hr × 63 min). Total experiment cost: **~$0.05**.

### Key Takeaways

- The ScaleEmbedding breaks through the DINO entropy wall, enabling much lower loss.
- View retrieval (same-image augmented view matching) is NOT the right metric for
  evaluating scale awareness — it needs a spacing-stratified evaluation.
- Next step: Implement cross-spacing retrieval eval (query with spacing-A, verify model
  distinguishes from spacing-B) and linear probe on organ classification.
- Both arms prove the full pipeline works: HF data hosting → cloud fetch → training →
  evaluation → result collection.

### Artifacts

- `runs/mvp-two-organ/baseline_train.log` — full training log
- `runs/mvp-two-organ/baseline_retrieval.json` — retrieval metrics
- `runs/mvp-two-organ/baseline_config.json` — full configuration
- `runs/mvp-two-organ/scale_aware_train.log` — full training log
- `runs/mvp-two-organ/scale_aware_retrieval.json` — retrieval metrics
- `runs/mvp-two-organ/scale_aware_config.json` — full configuration

---

## Pan-Organ Evaluation Protocol Validation (2026-04-21)

**Goal:** Validate the 6-metric evaluation protocol (`scripts/evaluate_panorgan.py`)
end-to-end on locally trained checkpoints.

### Training (local, RTX 2070 SUPER)

| Arm | Steps | Final Loss | Time | Hardware |
|-----|-------|------------|------|----------|
| Baseline | 1,000 | 9.017 | 2.5 min | RTX 2070 SUPER (8GB) |
| Scale-Aware | 1,000 | 8.920 | 2.5 min | RTX 2070 SUPER (8GB) |

Config: ViT-Small, batch=16×accum=4=eff64, lr=2e-4, warmup=100, AMP fp16.

### Evaluation Results (6 metrics, 4,035 val slices)

#### Metric 1: Per-Dataset View Retrieval

| Dataset | Baseline | Scale-Aware |
|---------|----------|-------------|
| LIDC-IDRI top-1 ratio | 1.0× | **8.0×** |
| Pancreas-CT top-1 ratio | 3.0× | **6.0×** |

Scale-aware model achieves 4-8× better retrieval even at just 1K steps.

#### Metric 2: Dataset Discrimination Linear Probe

| Metric | Baseline | Scale-Aware |
|--------|----------|-------------|
| Accuracy | 0.523 | 0.523 |
| AUC | 0.880 | **1.000** |

Both models near chance on raw accuracy (models undertrained at 1K steps), but
the scale-aware model achieves **perfect AUC=1.0** — its representations fully
separate the two datasets in probability space.

#### Metric 3: Spacing Counterfactual Test

| Intervention | Scale-Aware Distance |
|-------------|---------------------|
| Real → 2× spacing | 0.0003 |
| Real → ½× spacing | 0.0006 |
| ½× → 2× spacing | — |

Small but nonzero distances after only 1K steps (ScaleEmbedding is zero-initialized
and barely activated). Expect much larger distances with 5K+ step checkpoints.
Baseline: N/A (no scale embedding).

#### Metric 4: Domain Clustering

| Metric | Baseline | Scale-Aware |
|--------|----------|-------------|
| Same-dataset NN rate | 0.992 | 1.000 |
| Expected random | 0.527 | 0.527 |
| Enrichment | 1.9× | 1.9× |

Both models strongly cluster by dataset. Expected — these are different anatomies.

#### Metric 5: Spacing Prediction (Sanity Check)

| Metric | Baseline | Scale-Aware |
|--------|----------|-------------|
| R² (log spacing_x) | -0.005 | **0.724** |
| MAE (log spacing) | 0.109 | 0.050 |

**Key finding:** Baseline R²≈0 (cannot predict spacing from features). Scale-aware
R²=0.724 — the ScaleEmbedding successfully encodes physical spacing into CLS tokens.
Partly circular (spacing is added to tokens), but confirms the plumbing works.

#### Metric 6: Embedding Statistics

| Metric | Baseline | Scale-Aware |
|--------|----------|-------------|
| LIDC embed StdDev | 0.0006 | **0.0050** |
| Pancreas embed StdDev | 0.0005 | 0.0008 |
| Cross-dataset centroid cos | 1.000 | 0.996 |
| PCA1-spacing corr (LIDC) | -0.204 | **0.540** |
| PCA1-spacing corr (Pancreas) | 0.168 | **0.993** |

The scale-aware model has 8× higher embedding diversity for LIDC-IDRI. The first
principal component of Pancreas-CT embeddings correlates 0.993 with pixel spacing —
near-perfect linear relationship between embedding structure and physical scale.

### Analysis

1. **Scale embedding works:** Spacing R²=0.724, PCA-spacing correlation up to 0.993,
   and perfect dataset AUC — all confirm the ScaleEmbedding successfully encodes
   physical dimensions into the representation space.

2. **Counterfactual distances small at 1K steps:** Expected behavior — the
   ScaleEmbedding is zero-initialized and gradually ramps up during training.
   At 5K steps (where loss=0.134), expect much larger counterfactual distances.

3. **Eval protocol validated:** All 6 metrics run successfully on 4,035 val slices
   in ~65 seconds on RTX 2070 SUPER. No errors, no edge cases.

4. **1K vs 5K steps:** These results complement the 5K-step cloud experiment.
   The scale-aware model already shows clear advantages at 1K steps, consistent
   with the dramatic loss improvement seen at 5K steps (0.134 vs 8.992).

### Artifacts

- `runs/eval_validation/baseline_eval.json` — full 6-metric baseline results
- `runs/eval_validation/scale_aware_eval.json` — full 6-metric scale-aware results

---

## Local Validation: 5K-Step Scale-Aware Training (2026-04-21)

**Goal:** Reproduce the cloud training result locally on consumer hardware (RTX 2070
SUPER, 8GB VRAM) and run the full 6-metric evaluation protocol on a properly trained
checkpoint.

### Key Finding: Gradient Accumulation Step Counting

Multiple prior local training attempts failed (DINO collapse or flat loss) due to a
critical step-counting issue: `phase5_big_run.py` counts **micro-batches**, not
**optimizer steps**. With high accumulation factors (e.g., accum=32 for small GPUs),
the model receives far fewer optimizer updates than expected, and the LR schedule
compresses warmup to too few optimizer steps.

**Fix:** Use accum=4 (same as the successful cloud run) with batch_size=8. This yields
5000/4=1250 optimizer updates and 500/4=125 warmup optimizer updates — identical to
the cloud run's schedule. The effective batch is smaller (32 vs 256) but DINO still
converges.

### Training Configuration

- **Model:** ViT-Small (dim=384, depth=12, heads=6, patch=14, 21.7M backbone params)
- **Effective batch:** 32 (batch=8 × accum=4) — smaller than cloud's 256
- **Optimizer:** AdamW, lr=2e-4, cosine decay to 1e-6, warmup=500 micro-steps
- **Loss:** DINO + Gram(1.0) + KoLeo(0.1)
- **EMA:** 0.996, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
- **Steps:** 5,000 micro-batches (1,250 optimizer updates)
- **AMP:** disabled (not needed, batch_size=8 fits in 8GB)
- **Seed:** 42
- **Hardware:** NVIDIA GeForce RTX 2070 SUPER (8GB)
- **Git commit:** `984d229506f10a4b79fe901a1b1576e5897b51c9`
- **Architecture:** `zoo.arch.PatchViT` (timm-style attention/MLP)

### Loss Curve

| Step | Optimizer Update | Loss | LR |
|------|-----------------|------|-----|
| 0 | 0 | 8.97 | 4e-7 |
| 500 | 125 | 8.94 | 2e-4 (peak) |
| 1000 | 250 | 8.30 | 1.95e-4 |
| 2000 | 500 | 6.27 | 1.50e-4 |
| 3000 | 750 | 4.31 | 8.1e-5 |
| 4000 | 1000 | 1.25 | 2.4e-5 |
| 5000 | 1250 | 1.03 | 1.0e-6 |

Total training time: **733 seconds (~12 minutes)** at 6.82 steps/s, 218 samples/s.

### Evaluation Results (6 metrics, 3,843 val slices)

#### Metric 1: Per-Dataset View Retrieval

| Dataset | 1K steps (prior) | 5K steps (this run) |
|---------|-----------------|---------------------|
| LIDC-IDRI top-1 ratio | 8.0× | **14.0×** |
| Pancreas-CT top-1 ratio | 6.0× | **5.0×** |

#### Metric 2: Dataset Discrimination

| Metric | 1K steps | 5K steps |
|--------|----------|----------|
| Accuracy | 0.523 | **1.000** |
| AUC | 1.000 | **1.000** |

Perfect discrimination — model fully separates lung CT from abdominal CT.

#### Metric 3: Spacing Counterfactual

| Intervention | 1K steps | 5K steps |
|-------------|----------|----------|
| Real → 2× spacing | 0.0003 | **0.0551** |
| Real → ½× spacing | 0.0006 | **0.1072** |

**184× improvement** in spacing sensitivity. The model now strongly encodes
physical scale into its representations.

#### Metric 4: Domain Clustering

| Metric | 1K steps | 5K steps |
|--------|----------|----------|
| Same-dataset NN rate | 1.000 | **0.998** |
| Enrichment vs random | 1.9× | **2.0×** |

#### Metric 5: Spacing Prediction

| Metric | 1K steps | 5K steps |
|--------|----------|----------|
| R² (log spacing_x) | 0.724 | **0.876** |
| MAE (log spacing) | 0.050 | **0.038** |

#### Metric 6: Embedding Statistics

| Metric | 1K steps | 5K steps |
|--------|----------|----------|
| LIDC embed StdDev | 0.005 | **0.021** (4×) |
| Pancreas embed StdDev | 0.0008 | **0.020** (25×) |
| Cross-centroid cosine | 0.996 | **0.164** |
| PCA1-spacing corr (LIDC) | 0.540 | -0.198 |
| PCA1-spacing corr (Pancreas) | 0.993 | **-0.999** |

**Cross-dataset centroid cosine dropped from 0.996 to 0.164** — the model now
creates genuinely distinct embedding regions for different organs. Embedding diversity
improved 4-25× across both datasets.

### Comparison: Cloud (5K) vs Local (5K)

| Metric | Cloud (RTX 5070) | Local (RTX 2070) |
|--------|-----------------|-----------------|
| Final loss | 0.134 | 1.03 |
| Effective batch | 256 | 32 |
| View retrieval ratio | 4.0× | 14.0× (LIDC) |
| Spacing R² | — | 0.876 |
| Training time | 63 min | 12 min |

The local model has higher final loss (1.03 vs 0.134), likely due to smaller effective
batch. However, the local evaluation metrics are strong — the model has learned
meaningful scale-aware representations.

### Analysis

1. **Step counting matters:** The gradient accumulation step counter bug was the root
   cause of all prior local training failures. With accum=4 (matching the cloud run's
   optimizer-step schedule), training converges smoothly on consumer hardware.

2. **Small effective batch works:** Effective batch=32 (8× smaller than cloud's 256) still
   produces a well-trained model. DINO benefits from large batches for contrastive learning,
   but 32 is sufficient for convergence on this dataset size.

3. **Scale awareness validated end-to-end:** The spacing counterfactual test shows 184×
   improvement over the 1K-step model. The model genuinely distinguishes physical scales.

4. **Hub-ready:** The trained backbone (21.7M params, 86.7MB) has been exported to
   hub format (`runs/v1-local-5k/hub-export/`).

### Artifacts

- `runs/v1-local-5k/20260421_202330/checkpoint_final_00005000.pth` — final checkpoint
- `runs/v1-local-5k/eval_5k.json` — full 6-metric evaluation results
- `runs/v1-local-5k/hub-export/` — hub-format backbone + config

---

## Historical Experiments (Single-Dataset LIDC-IDRI)

| Run ID | Model | Eff Batch | LR | Warmup | T-Temp | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20260109_162920_4090_512px_DeepFreeze` | vit-large (p14) | 256 | 2e-05 | 3000 | 0.02 | Running |  |
| `20260109_104007_4090_LowLR_IceAge` | vit-large (p14) | 256 | 5e-05 | 1000 | 0.02 | Completed | Success. Best stability. Found Golden Zone LR ~2e-5. Drifted up at 5e-5. |
| `20260108_203723_4090_IceAge` | vit-large (p14) | 256 | 0.0002 | 1000 | 0.02 | Completed | Success! Broken entropy wall (6.78). Frozen Teacher (0.9995) + Sharp Temp (0.02). |
| `20260108_175149_4090_dim2048_test` | vit-large (p14) | 256 | 0.0002 | 1000 | 0.04 | Completed |  |
| `20260108_084549_4090_224px_slower` | vit-large (p14) | 256 | 0.0002 | 5000 | 0.05 | Completed |  |
| `20260108_003002_amd395_giant_prod_run` | vit-large (p14) | 256 | 0.0002 | 5000 | 0.04 | Completed |  |
| `20260107_211157_4090_224px` | vit-large (p14) | 256 | 0.0005 | 2500 | 0.05 | Stopped | Failed. Flatlined at 9.01. LR 5e-4 too high. |
| `20260107_181724_4090_warm_up_test` | vit-large (p16) | 256 | 0.0001 | 200 | 0.03 | Completed | Success/Stopped. Proof of throughput. Hit 9.01 wall. Backbone good. |
| `20260107_181716_4090_warm_up_test` | vit-large (p16) | 256 | 0.0001 | 2500 | 0.03 | Stopped |  |
| `20260106_223221_4090_large_teacher0-015` | vit-large (p16) | 256 | 0.0005 |  | 0.015 | Completed |  |
| `20260104_194515_4090_large_teacher0-03` | vit-large (p16) | 256 | 0.0001 |  | 0.03 | Stopped |  |
| `20260104_181120_amd395_giant_64x4` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Completed |  |
| `20260104_170333_amd395_giant_128x2` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Stopped |  |
| `20260104_165932_amd395_giant_128x2` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Stopped |  |
| `20260104_145936` | vit-large (p14) | 256 | 0.0005 |  | 0.04 | Stopped |  |
| `20260104_123726_4090_large_teacher0-025` | vit-large (p16) | 256 | 0.0005 |  | 0.025 | Completed |  |
| `20260104_110654_4090_large_teacher0-02` | vit-large (p16) | 256 | 0.0005 |  | 0.02 | Completed |  |
| `20260104_093402` | vit-large (p16) | 256 | 0.0005 |  | 0.04 | Completed |  |
| `20260103_175300` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_165249` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_163425` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_162833` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_155511` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_155138` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_155103` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_155028` | vit-large (p14) | 288 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154819` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154759` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154642` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_152502` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_150009` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_144457` | vit-large (p14) | 16 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_144115` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143710` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143550` | vit-large (p14) | 192 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143247` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_141953` | vit-large (p14) | 32 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_141940` | vit-large (p14) | 32 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_141803` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_141734` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Stopped |  |

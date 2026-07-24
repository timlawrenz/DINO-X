# Discontinuation Notice — 5-Dataset Phase 5 ViT-Large 50K Pretraining

**Date:** 2026-07-22 · **Arm:** `5dataset-phase5-large-bs256-v2` · **Verdict:** KILL

---

## What We Tried

ViT-Large (923M params, dim=1024, depth=24, heads=16) with scale-aware embedding on the 5-dataset pan-organ CT corpus (400K slices: cq500, lidc-idri, msd-colon, msd-hepatic-vessel, pancreas-ct). DINO + KoLeo(0.1) + z-stride=3 + diverse batches. Resumed from step 5,000 on 2026-07-19 after memory mitigation fix (`expandable_segments:True` + batch 4×64 + grad checkpointing). Ran through step 50,000.

## What Happened

| Phase | Steps | Teacher Entropy | View Retrieval | What It Means |
|---|---|---|---|---|
| Stable learning | 5K–25K | 7.5 → 6.2 | 30× → **34× peak** | Genuine representation learning, pre-registered gate met (entropy ≥ 6.0) |
| Plateau | 25K–35K | 6.2 ↔ 4.3 (cycles) | 34× → 32× | Diminishing returns — each sharpen-expand cycle stopped producing gains |
| Pre-collapse chaos | 35K–40K | 0.81 ↔ 3.48 (step-to-step) | — | Adjacent-step entropy swings > 2.0; LR at 3e-6 too low to dampen |
| Zombie regime | 40K–50K | 2.5 mean, still oscillating | — | Model stopped learning, entropy never recovered to healthy range |

## Why It Failed

**Root cause:** The DINO + KoLeo objective on this 5-dataset corpus rewards scanner/hospital fingerprinting and spacing geometry over cross-organ feature learning. The 6-metric pan-organ eval at step 35K revealed:

1. **Scanner fingerprinting dominates** — Dataset discrimination AUC 0.978 (at 35K), rising to 0.981 at 50K. The model can almost perfectly identify which hospital acquired a scan, regardless of anatomy.

2. **Capacity dilution** — The two smallest datasets (msd-colon 3,688 slices, msd-hepatic-vessel 4,093 slices) collapsed into near-identical embedding space (cosine 0.991 at 35K, 0.962 at 50K). The model abandoned minority organs.

3. **Spacing geometry works, but doesn't help** — Spacing prediction R² 0.980–0.981, counterfactual distances 0.31–0.44. Scale embedding succeeded at what it was designed for, but this competency occupies embedding dimensions at the expense of anatomy.

4. **Objective deadlock** — Between spacing geometry (concrete signal, easy gradient) and scanner fingerprinting (dataset-level pattern, strong gradient), the model had no incentive to learn cross-organ anatomy (weak gradient, no explicit loss term). Entropy oscillations at 40K were the model cycling between these easier objectives without finding the harder one.

## Key Evidence (Pre-Registered Gate Failure)

**Pre-registered gate:** PASS if teacher entropy ≥ 6.0 for 10K steps AND view retrieval > 100×.  
**Actual:** View retrieval peaked at **34×** (step 25K), well below 100× gate.

| Checkpoint | View Retrieval | Dataset AUC | Cross-Dataset Collapse |
|---|---|---|---|
| 35K (pre-collapse) | 32× aggregate, 2/5 datasets pass | 0.978 | colon↔vessel 0.991 |
| 50K (zombie) | 32× aggregate, 2/5 datasets pass | 0.981 | colon↔vessel 0.962 |

## What Survives

1. **Scale-aware embedding is proven** — R² 0.980–0.981 on spacing prediction, counterfactual distances 0.31–0.44. This should be carried forward into any successor architecture.

2. **Memory mitigation recipe** — `expandable_segments:True` + batch 4×64 + grad checkpointing is the solved config for >900M-param models on Strix Halo. Throughput improved from 27 → 46 img/s as a side effect.

3. **Pan-organ 6-metric eval** — The eval suite correctly diagnosed scanner fingerprinting and capacity dilution that view retrieval alone missed. Should be standard for any multi-dataset pretraining effort.

4. **Best checkpoint: step 25K** — Peak view retrieval (34×), pre-chaos entropy (6.19). Usable for spacing-sensitive tasks but not organ-generalizable.

5. **MC Dropout uncertainty** — The arXiv paper (2607.16317) confirms that deterministic entropy estimators (both DINO's teacher and auxiliary confidence heads) collapse by construction. Any successor should use sampling-based (MC Dropout) uncertainty, not softmax entropy.

## What Changes Next Time

1. **Change the objective** — DINO's cross-entropy on softmax outputs inherently rewards collapse into dataset-specific features. Alternatives: MAE (masked image modeling), adversarial domain adaptation (gradient reversal on dataset ID), or organ classification as auxiliary loss.

2. **Balance the corpus** — The two smallest datasets (3,688 and 4,093 slices) were abandoned. Either drop them, oversample them, or pre-train on balanced organ-specific subsets.

3. **Drop teacher entropy as a health metric** — It's a convergence measure, not an uncertainty measure. It naturally trends toward 0 and oscillates. MC-Dropout predictive entropy doesn't collapse.

4. **Consider pre-training on non-medical data** — Starting from DINOv2/ImageNet weights might break the scanner-fingerprinting deadlock by providing a feature space not dominated by hospital-specific patterns.

## Artifacts Preserved

- `runs/20260719_042301_5dataset-phase5-large-bs256-v2/` — full run with checkpoints at 10K, 15K, 20K, 25K, 30K, 35K, 40K, 45K, 50K, and final
- `results/panorgan_step35000.json` — pre-collapse pan-organ eval
- `results/panorgan_step50000.json` — zombie-phase pan-organ eval
- `runs/20260719_042301_5dataset-phase5-large-bs256-v2/view_retrieval_step*_N512.json` — view retrieval trend at 15K, 20K, 25K, 30K, 35K

## Do Not Re-Run

This architecture + objective + dataset combination is definitively disproven. Do not resume from any checkpoint, do not re-run with different hyperparameters on this recipe, and do not scale up dataset size without changing the training objective. The problem is structural, not configurational.

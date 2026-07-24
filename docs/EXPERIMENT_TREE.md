# Experiment Tree

Living workstream map — status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`. Evidence lives in the ledger (`docs/EXPERIMENTS_AND_RESULTS.md`). Governance: `docs/experiment-structure.md`.

---

## Active

_No active experiments._

## Concluded

- **[CONCLUDED — PIVOT] Replace View Retrieval Gate for Single-Organ Models** (`replace-view-retrieval-gate`)
  - Tested spacing metrics as replacement gates on LIDC specialist (known good AUROC 0.728).
  - Failed targets (counterfactual 0.239 < 0.30, R² 0.941 < 0.95). Single datasets lack spacing variance to drive scale embeddings to pan-organ levels.
  - **Verdict:** Spacing gates invalid for single datasets. View retrieval invalid (architecture handicapped).
  - **New Governance:** Single-organ models gated **solely by LoRA AUROC** vs baseline.

- **[CONCLUDED — KILL] Positional Bias Projection for View Retrieval** (`positional-bias-projection`)
  - Noise-image PCA bias projection applied to layer 9 features on 50K checkpoint.
  - Result: **37×** — identical to layer 9 without bias projection. No improvement.
  - Hypothesis that DINO positional artifact degrades CLS-token view retrieval was disproven for LIDC CT.
  - Added `--pos-bias-project` flag to eval script.

- **[CONCLUDED — KILL] Phase 5: ViT-Large Pan-Organ Pretraining** (`runs/20260719_042301_5dataset-phase5-large-bs256-v2/`)
  - 5-dataset pan-organ corpus (400K slices), ViT-Large (923M params), effective batch 256
  - Resumed 2026-07-19, completed 2026-07-22 at step 50,000. Memory mitigation stable; run finished cleanly.
  - View retrieval peaked at 34× (step 25K) vs pre-registered 100× gate. Entropy collapsed to chaotic oscillation at 40K. Pan-organ eval confirmed scanner fingerprinting (AUC 0.981) and capacity dilution (colon↔vessel cosine 0.962).
  - **Root cause:** DINO + KoLeo objective rewards scanner identity and spacing geometry over cross-organ anatomy. ViT-Small and ViT-Large both hit same wall at different speeds — model size is not the bottleneck.
  - See `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`.

- **[CONCLUDED — GO] MVP Two-Organ Scale-Aware Ablation** (`runs/mvp-two-organ/`)
  - Baseline (no scale) vs Scale-Aware ViT-Small, 43K slices (LIDC + Pancreas-CT), 5K steps
  - Scale-Aware: 67× lower loss (0.134 vs 8.992), both arms healthy. View retrieval 5.0× vs 4.0×.

- **[CONCLUDED — GO] Pan-Organ Evaluation Protocol Validation** (`runs/eval_validation/`)
  - 6-metric eval suite validated on 1K-step checkpoints
  - Scale-aware: R²=0.724 spacing prediction, AUC=1.000 dataset discrimination

- **[CONCLUDED — GO] Local 5K-Step Scale-Aware** (`runs/v1-local-5k/`)
  - ViT-Small on RTX 2070 SUPER, 12 min training
  - View retrieval 14× (LIDC), spacing R²=0.876, cross-centroid cosine 0.164, 184× spacing sensitivity improvement

- **[CONCLUDED — GO] 4-Dataset Anti-Memorization Ablation** (`runs/vit-small-4dataset-ablation/`)
  - Fixed catastrophic collapse with KoLeo(0.1) + crop=0.3 + z-stride=3 + diverse batches
  - View retrieval 7.0× at 5K steps. Cleared architecture for pan-organ scaling.

- **[CONCLUDED — GO] LIDC Malignancy LoRA Benchmark** (`adapters/lidc-malignancy-lora-r8-64px-lung-window/`)
  - 4-dataset backbone, LoRA rank=8, 64px crops, lung HU window → AUROC 0.710
  - Crop size dominates (64px >> 128px >> 224px). Lung window adds +3% vs generic.

- **[CONCLUDED — GO] 5-Dataset Phase 3 Pretraining** (`runs/20260422_202622_5dataset-phase3-small/`)
  - 400K slices, ViT-Small, 50K steps, effective batch 128. View retrieval 56×.
  - Training stable with temperature-scaled sampling (T=2.0). bfloat16, no NaN events.

- **[CONCLUDED — GO] LoRA Benchmark: 5-Dataset vs 4-Dataset** (`adapters/lidc-malignancy-5dataset-*`)
  - 5-dataset AUROC 0.680 vs 4-dataset 0.710. Capacity dilution confirmed.
  - More pretraining steps degraded lung-specific features (10K > 25K > 50K).

- **[CONCLUDED — GO] 5-Dataset Doubled Batch Size (bs256)** (`runs/20260423_171906_5dataset-phase3-small-bs256/`)
  - Effective batch 128→256. Loss plateaued at ~35K. LoRA AUROC 0.684, view retrieval 54×.
  - Doubling batch did not close the 0.026 gap to 4-dataset specialist.

- **[CONCLUDED — GO] Ablation Scan: 5-Dataset LoRA Interventions** (`adapters/ablation-5dataset-bs256-*`)
  - Rank 16: 0.685 AUROC (best). Rank 32: overfits. Unfreeze 1 block: hurts (-0.009). 128px crops: -0.079.
  - No LoRA-side intervention bridges the gap. The gap is structural (backbone capacity).

- **[CONCLUDED — KILL] ViT-Small 100K Extended Pretraining** (`runs/20260428_*_5dataset-phase3-small-bs256-100k/`)
  - 5-dataset, cosine schedule restretched to 100K. 81K: AUROC 0.697 (climbing), view retrieval 63×.
  - 86K: catastrophic capacity collapse (teacher entropy 0.001, AUROC 0.642).
  - ViT-Small definitively saturated. Path forward: ViT-Base or ViT-Large.
  - See `docs/DISCONTINUATION_NOTICE_vit-small-100k.md` (to be created).

### Historical (Single-Dataset LIDC-IDRI, Jan 2026)

- **[CONCLUDED — GO] Ice Age / Low LR experiments** (`20260108_203723_4090_IceAge`, `20260109_104007_4090_LowLR_IceAge`)
  - ViT-Large, LIDC-only, effective batch 256. Found "Golden Zone" LR ~2e-5. Broken entropy wall (loss 6.78) with frozen teacher (0.9995) + sharp temp (0.02).
- Multiple additional ViT-Large single-dataset runs (Jan 3–9, 2026) — most Completed, some Stopped. See ledger for full table.

## TBD

- **[TBD] Positional Bias Projection for View Retrieval** — apply arXiv:2604.23670 null-space projection to push view retrieval from 37× past the 40× gate on the existing 50K checkpoint. Low risk, evaluation-only change.
- **[TBD] Replace View Retrieval Gate for Single-Organ Models** — layer-selection confirmed view retrieval is architecture-handicapped (31→37×). Use pan-organ metrics (spacing counterfactual, dataset discrimination) or LoRA AUROC as the primary gate for single-organ specialists.
- **[TBD] Scale LIDC Recipe to Other Organs** — apply the proven ViT-Base + scale-aware + single-organ recipe to cq500 (brain), pancreas-ct, msd-colon, msd-hepatic-vessel. Baseline AUROC 0.728 on LIDC shows the approach works.
- **[TBD] Fix `--ckpt-keep-last` Default** — change from 5 to 0 (keep all) or 10 to prevent checkpoint rotation from destroying intermediate evaluation points (lost 25K in Phase 6).
- **[TBD] Many-to-Many View Retrieval Matching** — relax 1-to-1 nearest neighbor to top-K mutual neighbors (arXiv:2604.23670) in `phase5_view_retrieval_eval.py`.
- **[TBD] Cross-Modality Expansion** — MRI (`dinox-mri-vit-small`) and X-ray (`dinox-xray-vit-small`) models.
- **[TBD] HF Hub Release** — publish `dinox-ct-vit-small-v1` model card + safetensors.
- **[TBD] Adversarial Pass Retrospective** — audit all GO verdicts against the 4-question checklist.
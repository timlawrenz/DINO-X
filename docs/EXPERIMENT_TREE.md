# Experiment Tree

Living workstream map — status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`. Evidence lives in the ledger (`docs/EXPERIMENTS_AND_RESULTS.md`). Governance: `docs/experiment-structure.md`.

---

## Active

- **[ACTIVE — STALLED] Phase 5: ViT-Large Pan-Organ Pretraining** (`runs/20260511_032957_5dataset-phase5-large-bs256/`)
  - 5-dataset pan-organ corpus (400K slices), ViT-Large (923M params), effective batch 256
  - 4 aborts on Strix Halo (May 10–11) — ROCm allocator fragmentation. Model was learning well (teacher entropy 6.60 at step 6,369). Checkpoints at step 5,000 and 6,369 available.
  - **Resume plan:** `runs/20260511_032957_5dataset-phase5-large-bs256/README_RESUME.md` — memory mitigation with `expandable_segments:True` + smaller physical batch (4×64)
  - See also: `docs/phase6_large_model_resume.md`

## Concluded

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

- **[TBD] Resume ViT-Large Pan-Organ** — memory mitigation + validation. See PROJECT_STATUS.md.
- **[TBD] ViT-Base Pan-Organ** — intermediate scaling step if ViT-Large proves unstable.
- **[TBD] Cross-Modality Expansion** — MRI (`dinox-mri-vit-small`) and X-ray (`dinox-xray-vit-small`) models.
- **[TBD] Attention Map Evaluation** — unsupervised nodule segmentation from attention maps.
- **[TBD] Linear Probe AUC > 0.90** — on LIDC malignancy (Stage C evaluation).
- **[TBD] Per-Class Accuracy Logging** — needed for rare-class recall evaluation.
- **[TBD] HF Hub Release** — publish `dinox-ct-vit-small-v1` model card + safetensors.
- **[TBD] Adversarial Pass Retrospective** — audit all GO verdicts against the 4-question checklist.
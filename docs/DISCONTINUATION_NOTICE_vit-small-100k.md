# Project Discontinuation Notice — ViT-Small 100K Extended Pretraining

**Date:** 2026-07-15
**Project:** DINO-X — ViT-Small 5-dataset pan-organ pretraining extended to 100K steps
**Status:** DISCONTINUED (KILL)

## Summary

Extended pretraining of ViT-Small (dim=384, 22M backbone params) on the 5-dataset pan-organ corpus from 50K to 100K steps, restretching the cosine LR schedule. The model showed promising improvement at 81K steps (LoRA AUROC 0.697, view retrieval 63×) but catastrophically collapsed by 86K steps (teacher entropy 0.001, AUROC 0.642). ViT-Small capacity is definitively saturated on 5 heterogeneous organ domains.

## What We Learned

### Successful components ✅

- **Extended cosine schedule can recover organ-specific features.** At 81K steps, LoRA AUROC climbed from 0.684 (50K) to 0.697 — the highest 5-dataset result ever. The warm restart from 50K (LR at cosine midpoint ~1e-4) gave the model a second wind.
- **View retrieval improved throughout.** 54× (50K) → 63× (81K) → 61× (86K). The model continued learning general-purpose representations even as organ-specific features degraded.
- **Temperature-scaled sampling (T=2.0) held stable.** No dataset was starved or dominated. The 5-dataset mix remained balanced across 100K steps.

### Failed components ❌

- **Catastrophic capacity collapse at 86K.** Teacher entropy dropped from ~0.04 (81K estimate) to 0.001 at 86K. Student entropy: 0.00018. The model produced near-zero-entropy predictions — effectively a constant output.
- **LoRA AUROC plummeted by 0.055.** 0.697 (81K, best epoch 33) → 0.642 (86K, best epoch 13). This was not noise — the backbone's feature space collapsed.
- **Collapse was not detected in time.** The model ran for 5K additional steps (81K → 86K) with collapsed features before being manually evaluated. View retrieval at 86K was still 61×, masking the collapse. View retrieval measures coarse similarity, not feature quality.

## Root Cause

**ViT-Small (22M backbone params) cannot maintain organ-specific feature diversity across 5 heterogeneous organ domains under DINO's contrastive pressure.** The model is forced to compress representations into a space that optimizes DINO loss, which favors global consistency over local organ specificity. At 81K steps, the cosine LR provided a temporary reprieve (higher LR = more exploration), but as LR decayed, the model re-collapsed into the same entropy floor.

This is a **structural capacity ceiling**, not a hyperparameter issue. Evidence:
- 4 learning rates tested (1e-4 through 1e-3) — none bridged the gap
- 2 effective batch sizes tested (128, 256) — identical results
- 6 checkpoints across 2 batch sizes — consistent ~0.026 gap to 4-dataset specialist
- 5 LoRA interventions (rank 8/16/32, unfreeze 1 block, 128px crops) — none closed the gap
- The 81K→86K collapse is the final proof: even extended training doesn't help when capacity is the bottleneck

The analogy: a 22M-parameter model trying to represent lung, liver, colon, brain, and pancreas simultaneously is like trying to fit a symphony orchestra into a telephone booth.

## Why We're Sharing This

**Negative results are valuable.** A future researcher seeing this repo should NOT:
- Train ViT-Small on 5+ organ datasets and expect organ-specific features
- Interpret view retrieval ratio as a proxy for downstream task fitness (63× retrieval ≠ 0.697 AUROC)
- Extend cosine schedules on capacity-limited models expecting monotonic improvement
- Trust loss/retrieval metrics without monitoring teacher entropy for collapse

## Salvage

| Artifact | Where | Value |
|---|---|---|
| 81K checkpoint | `runs/20260428_*_5dataset-phase3-small-bs256-100k/checkpoint_00080000.pth` | Best ViT-Small 5-dataset backbone (AUROC 0.697, view retrieval 63×). Worth keeping as ViT-Small reference. |
| 50K bs256 checkpoint | `runs/20260423_171906_5dataset-phase3-small-bs256/checkpoint_final_00050000.pth` | Stable baseline for LoRA comparisons. |
| All adapters | `adapters/lidc-malignancy-5dataset-bs256-*` | Full ablation scan results — baseline, rank 16/32, unfreeze, 128px crops. |
| Temperature-scaled index | `data/mvp/combined_5dataset_t2.csv` | Reusable for ViT-Base or ViT-Large training. |
| Split manifest | `data/mvp/split_manifest_5dataset.json` | Train/val split, reusable. |
| Training logs | `runs/20260428_*_5dataset-phase3-small-bs256-100k/events.out.tfevents.*` | TensorBoard data — shows collapse trajectory. |

## Path Forward

The successor is **Phase 5: ViT-Large pan-organ pretraining** (923M params, dim=1024, 24 layers). The large model showed no sign of collapse (teacher entropy climbing toward 6.78 at step 6,369) before Strix Halo memory crashes halted it. See `PROJECT_STATUS.md` for resume plan.

If ViT-Large proves too unstable on Strix Halo, **ViT-Base** (dim=768, 12 layers, ~86M backbone) is the intermediate scaling step — enough capacity to escape the ViT-Small ceiling without the memory pressure of ViT-Large.
# 02 — Training Recipe

Exactly what went into this model. Every config value is sourced from the frozen
`config.json` at run time (immutable) and `finetune_config.json`, not from memory.

---

## 1. Self-supervised pretraining (the backbone)

**Dataset:** MSD Hepatic-Vessel (Medical Segmentation Decathlon Task08), single-organ
only — 48,021 slices. This avoids the pan-organ capacity dilution.

**Architecture:** ViT-Base, scale-aware.

| Param | Value |
|---|---|
| patch | 14 |
| dim | 768 |
| depth | 12 |
| heads | 12 |
| mlp_ratio | 4.0 |
| out_dim | 8192 |
| img_size | 224 |

**Scale awareness:** `ScaleEmbedding` injects DICOM spacing (`spacing_x/y/z`) into the
patch embeddings, so the model knows physical dimensions (7mm vs 21mm of tissue), matching
how radiologists read. `scale_aware: true`.

**DINO self-supervised objective + anti-collapse regularizers:**

| Param | Value |
|---|---|
| loss_type | dino |
| gram_enabled / weight | true / 1.0 |
| koleo_weight | 0.1 |
| lr | 2e-5 |
| min_lr | 1e-6 |
| warmup_steps | 2500 |
| weight_decay | 0.04 |
| max_steps | 50000 |
| ema | 0.996 |
| teacher_temp | 0.04 |
| student_temp | 0.1 |
| center_momentum | 0.999 |
| batch_size / accumulation | 4 / 64 (effective 256) |
| data augmentation (HU window) | random level [-400,400], width [800,2000] |

**Hardware:** max395, Radeon 8060S (Strix Halo / ROCm), bfloat16 AMP. Throughput ~305 img/s,
50K steps in ~11.7h. No entropy collapse (healthy training per ledger Phase 9).

## 2. Preprocessing

DICOM CT volumes → 16-bit lossless PNG per axial slice, HU clipped to [-1000, 4000],
offset +32768 (the project's `hu16_png` format). Index CSV carries spacing metadata.
Preprocessing script: `scripts/preprocessing/phase2_preprocess_nifti.py` (MSD) /
`phase2_preprocess_lidc_idri.py` (DICOM).

## 3. LoRA fine-tuning (the task head)

**Task:** binary classification — is this slice hepatic/portal **vessel tissue** present?
(Labels from MSD Task08 segmentation masks; label=1 = vessel present.)

**Config (`adapters/msd-hepatic-vessel-vit-base-50k/finetune_config.json`):**

| Param | Value |
|---|---|
| task | classification |
| num_classes | 2 |
| rank | 8 |
| alpha | 16.0 |
| lr | 5e-4 |
| epochs | 50 |
| batch_size | 32 |
| input_format | hu16_png |
| HU window | level -30, width 120 |
| scale_aware | true (spacing passed) |
| seed | 42 |
| early stopping | AUROC, patience 10 |

**Result:** best internal AUROC **0.9456** at epoch 23 (best val loss 0.3236).

Script: `scripts/finetune_lora.py`. The backbone stays frozen; only LoRA + head are trained
(~5MB adapter). Task head saved separately (`head.pth`).

---

### Evidence pointers
- Backbone run config (frozen): `runs/20260728_180949_hepatic-specialist-vit-base-scale-aware/config.json`
- Adapter config: `adapters/msd-hepatic-vessel-vit-base-50k/finetune_config.json`
- Pretraining / fine-tuning scripts: `scripts/phase5_big_run.py`, `scripts/finetune_lora.py`
- Label extractor: `scripts/preprocessing/extract_msd_labels.py`
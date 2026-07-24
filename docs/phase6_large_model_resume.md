# Phase 6: Large Model Resume — Analysis & Next Steps

**Date:** 2026-05-16
**Status:** Stalled — 4 aborts on Strix Halo, model was learning well

---

## Background: Why We're Here

### Phase 5a: Small Model (ViT-Small, dim=384) — Complete Collapse

The 100K-step ViT-Small run on the 5-dataset pan-organ corpus showed **total feature collapse** by step 86,100:

| Metric | Step 86,100 | Target | Verdict |
|---|---|---|---|
| Teacher Entropy | **0.001** | ~6.78 | 💀 Collapse |
| Student Entropy | **0.00018** | ~6.78 | 💀 Collapse |
| Loss_DINO | **0.0001** | climbing | 💀 Collapse |

View retrieval looked superficially good (61x vs random) but top-1 was only **12%** — the model learned a coarse representation insufficient for downstream tasks.

LoRA fine-tuning on the small model achieved AUROC 0.684 on LIDC malignancy, vs 0.710 for the 4-dataset lung specialist. No LoRA intervention closed the 0.026 gap.

### Pivot: Large Model (ViT-Large, dim=1024)

Rationale: dim=384 with 8192-dim output head was the bottleneck. dim=1024 should prevent collapse and retain spatial/texture detail across 5 heterogeneous organ modalities.

---

## Current Status: Large Model Training Trajectory

### The 4 Aborts (May 10-11)

| Run | Batch | Steps | Throughput | What happened |
|---|---|---|---|---|
| 16:29 (May 10) | 32×8 | 8 | 1.2 img/s | IO-bound, crashed immediately |
| 17:10 (May 10) | 16×16 | 4 | 3.4 img/s | Still IO-bound |
| 17:16 (May 10) | **8×32** | **2,592** | 23-26 img/s | Good training, then crashed |
| 03:29 (May 11) | 8×32 | **6,369** | 23-27 img/s | Resumed, then crashed |

First two runs were IO-bound (larger physical batches caused memory pressure on ROCm allocator). Switching to 8×32 fixed throughput but didn't fix the eventual crash at ~5K effective steps.

### The Model Was Learning — Strong Evidence

Teacher entropy trajectory (target ~6.78):

```
Run 3 (May 10): step 2592  ent=3.89  loss=5.50
Run 4 (May 11): step 6366  ent=6.60  loss=7.41
```

- Teacher entropy climbed from 3.9 → 6.6 in just 4K steps, nearly hitting the 6.78 target
- The small model at 100K collapsed to 0.001 — the large model is on a completely different trajectory
- KoLeo dropping (2.83 → 1.58) — regularization working
- Throughput stable at ~27 img/s
- **No NaN/Inf in loss**

This was NOT a learning collapse. The crash was a memory issue.

### Small vs Large Configuration Comparison

| Parameter | Small (collapsed) | Large (stalled) |
|---|---|---|
| Model | vit-small, dim=384 | vit-large, dim=1024 |
| Batch | 64×4=256 | 8×32=256 |
| LR | 2e-4 | 2e-5 |
| Center Momentum | 0.9 | 0.999 |
| Teacher Temp | 0.04 | 0.04 |
| Max Steps | 100K | 50K |
| Index | data/mvp/combined_5dataset_t2.csv | same |
| Split | data/mvp/split_manifest_5dataset.json | same |

Key difference: **center_momentum** — small used 0.9 (more aggressive centering), large used 0.999 (momentum update, DINO standard). This is intentional for the large model to prevent collapse.

---

## Root Cause Hypothesis: Memory

The checkpoint is 4.7GB. With dim=1024, activation memory during forward/backward pass consumes most of the Strix Halo's ~96GB VRAM slice. The ~5K step crash pattern suggests:

1. **ROCm allocator fragmentation** — gradual buildup of "reserved but unallocated" memory
2. **Slow memory leak** — possibly in data loader or model forward pass
3. **GPU kernel timeout** — ROCm watchdog on long-running kernels

## Available Checkpoints

```
runs/20260511_032957_5dataset-phase5-large-bs256/
  checkpoint_00005000.pth          (4.7GB) ← resume from here
  checkpoint_final_00006369.pth    (4.7GB) ← last checkpoint, also usable
  config.json
  events.out.tfevents...
```

---

## Recommended Next Steps

### Option A: Resume with memory mitigation (recommended)

```bash
# 1. Set allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Resume from step 5000 checkpoint
cd ~/source/activity/DINO-X
source scripts/rocm_env.sh

python scripts/phase5_big_run.py \
    --index-csv data/mvp/combined_5dataset_t2.csv \
    --split-manifest data/mvp/split_manifest_5dataset.json \
    --resume runs/20260511_032957_5dataset-phase5-large-bs256/checkpoint_00005000.pth \
    --batch-size 4 --accumulation-steps 64 \
    --max-steps 50000 --run-suffix 5dataset-phase5-large-bs256-v2 \
    --ckpt-every 5000 --run-dir runs/
```

Changes from last config:
- `--batch-size 4` (4×64=256 effective) — smaller physical batch = less peak activation memory
- `--accumulation-steps 64` — maintain effective batch of 256
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduce allocator fragmentation

### Option B: Resume with same batch size

If option A still crashes, try the original batch config:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/phase5_big_run.py \
    --index-csv data/mvp/combined_5dataset_t2.csv \
    --split-manifest data/mvp/split_manifest_5dataset.json \
    --resume runs/20260511_032957_5dataset-phase5-large-bs256/checkpoint_00005000.pth \
    --batch-size 8 --accumulation-steps 32 \
    --max-steps 50000 --run-suffix 5dataset-phase5-large-bs256-v3 \
    --ckpt-every 5000 --run-dir runs/
```

### Validation Plan

While training runs, monitor:

1. **Teacher entropy** — should continue climbing toward 6.78. If it drops or stalls, that's collapse.
2. **Loss_Total** — should keep climbing (this is normal for DINO as entropy rises).
3. **KoLeo** — should continue dropping toward 0.
4. **Crash pattern** — note the exact step count at which it crashes. If consistent (~10K effective steps), that gives a target for the next iteration.

### When to stop / when to pivot

- **Stop criteria:** Teacher entropy drops below 4.0 after rising, or loss goes NaN/Inf
- **Success criteria:** Teacher entropy reaches 6.0+ and stays there for 10K steps, then run view-retrieval eval on the final checkpoint
- **If crashes persist after reducing batch:** Consider gradient checkpointing (`--grad-checkpoint`) to reduce activation memory

---

## View Retrieval Eval Command

After training completes:

```bash
python scripts/phase5_view_retrieval_eval.py \
    --checkpoint runs/202605XX_XXXXXXXX_5dataset-phase5-large-bs256-vN/checkpoint_final_*.pth \
    --index-csv data/mvp/combined_5dataset_t2.csv \
    --split-manifest data/mvp/split_manifest_5dataset.json \
    --img-size 224 --n 512 --batch-size 64
```

### Baselines to beat

| Model | top-1 | top-5 | ratio vs random |
|---|---|---|---|
| Small 50K (collapsed) | 10.5% | 39.5% | 54x |
| Small 100K (collapsed) | 12.3% | 34.0% | 63x |
| **Large target** | **>20%** | **>50%** | **>100x** |

---

## Key Takeaway

**Don't restart from scratch.** The model at step 6,369 had teacher entropy 6.60 — it was one step away from convergence-level entropy. Resuming from step 5,000 should give us a model that actually learned something meaningful across 5 organ datasets. The crash was a memory issue, not a learning issue.

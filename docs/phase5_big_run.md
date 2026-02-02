# Phase 5: Big Run - Production Training Guide

This guide covers executing Phase 5 production training for both validation (Phase 5a) and production (Phase 5b) runs.

## ⚠️ Critical: Gram Anchoring is Always Enabled

**Gram Anchoring is permanently enabled and cannot be disabled.** This is not a bug—it's a requirement. Without Gram Anchoring, DINO models trained on medical imaging (CT scans) will suffer immediate feature collapse, treating all lung tissue as identical. The model must preserve geometric texture correlations to distinguish nodules from vessels and other subtle tissue differences.

## Overview

Phase 5 implements a unified, parameterized training script (`scripts/phase5_big_run.py`) that supports:

- **Label-free validation (recommended):** run `scripts/phase5_view_retrieval_eval.py` on any checkpoint to measure whether two augmented views of the *same slice* retrieve each other in embedding space (Top-1/Top-k vs random baseline).

- **Phase 5a**: ViT-Large validation run (384 steps) on RTX 4090
- **Phase 5b**: ViT-Giant production run (15-day marathon) on AMD Strix Halo (amd395)
- Hardware-agnostic checkpoints (train on one GPU, resume on another)
- Automatic hardware detection and optimization
- Extended checkpoint management with rotation
- Training anomaly detection
- Full reproducibility and provenance tracking

## Quick Start

### Phase 5a: ViT-Large Validation Run (4090)

**Option 1: Standard Resolution (224px)**
Fastest for checking convergence mechanics.

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --max-steps 384 \
    --num-workers 8 \
    --batch-size 64 \
    --accumulation-steps 4 \
    --grad-checkpoint
```

**Option 2: Native Resolution (512px)**
Slower, but sees full texture detail. Requires `patch=16` for correct alignment.

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --vit-patch 16 \
    --img-size 512 \
    --max-steps 384 \
    --batch-size 16 \
    --accumulation-steps 16 \
    --grad-checkpoint
```

**Expected outcome:**
- Training completes in ~2-6 hours (depending on resolution)
- Loss should show decreasing trend
- Checkpoint saved to `data/runs/YYYYMMDD_HHMMSS/checkpoint_00000384.pth`
- Ready for Phase 6a validation

**Note:** `--grad-checkpoint` is **required** for ViT-Large on 24GB GPUs to fit larger batch sizes.

### Phase 5b: ViT-Giant Production Run (amd395)

```bash
python scripts/phase5_big_run.py \
    --config vit-giant \
    --batch-size 128 \
    --accumulation-steps 2 \
    --grad-checkpoint
```

**Expected outcome:**
- Training runs continuously until manually stopped (Ctrl+C)
- Utilizes >24GB unified memory on AMD Strix Halo
- Checkpoints saved every 100 steps (configurable with `--ckpt-every`)
- Automatic checkpoint rotation (keeps last 5 by default)

**Note:** Gradient checkpointing is strongly recommended even on large-memory systems to maximize batch size.

## Configuration Presets

The script includes two built-in model configurations:

### ViT-Large (for Phase 5a)
```
Patch size: 14
Dimension: 1024
Depth: 24 transformer blocks
Attention heads: 16
Parameters: ~304M
Memory requirement: ~20GB @ batch 64
```

### ViT-Giant (for Phase 5b)
```
Patch size: 14
Dimension: 1408
Depth: 40 transformer blocks
Attention heads: 16
Parameters: ~1.1B
Memory requirement: >24GB @ batch 128
```

## Hardware Detection

The script automatically detects hardware and applies optimizations:

### RTX 4090 (CUDA)
```
num_workers: 8
pin_memory: True
batch_size_recommendation: 64
```

### AMD Strix Halo (ROCm)
```
num_workers: 16
pin_memory: False (unified memory optimization)
batch_size_recommendation: 128
```

## Checkpoint Management

### Automatic Resume

Resume from the latest checkpoint in the most recent run:

```bash
python scripts/phase5_big_run.py --config vit-large --resume auto
```

### Resume from Specific Checkpoint

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --resume data/runs/20260103_120000/checkpoint_00000384.pth
```

### Cross-Hardware Resume

Train on 4090, then resume on amd395:

```bash
# On 4090: Train for 384 steps
python scripts/phase5_big_run.py --config vit-large --max-steps 384

# On amd395: Resume and continue training
python scripts/phase5_big_run.py \
    --config vit-large \
    --resume data/runs/20260103_120000/checkpoint_00000384.pth
```

The script will:
- ✅ Load model weights successfully
- ✅ Detect hardware change (CUDA → ROCm)
- ✅ Apply new optimization presets automatically
- ✅ Log the platform change

### Checkpoint Rotation

By default, only the last 5 checkpoints are kept to save disk space:

```bash
python scripts/phase5_big_run.py \
    --config vit-giant \
    --ckpt-keep-last 10  # Keep last 10 checkpoints
```

## Label-free Validation (No Labels Required)
After you have a checkpoint, you can run a **label-free view-retrieval evaluation** on the validation split:

```bash
python scripts/phase5_view_retrieval_eval.py \
  --checkpoint data/runs/YYYYMMDD_HHMMSS/checkpoint_00001000.pth \
  --split-manifest data/processed/_splits/val10_seed0.json \
  --index-csv data/processed/_index/index.csv \
  --n 4096
```

**Interpretation:**
- `top1` is retrieval accuracy (query=view1, keys=view2; correct key is same index).
- `baseline` is ~`1/n`.
- `ratio` is `top1 / baseline`.
- The script exits **0 on pass** and **2 on fail**, and writes a `view_retrieval_step*_N*.json` metrics file.

## Augmentation Strategy & Convergence

**Critical Finding (2026-02-01):**
Early attempts at training DINOv3 on LIDC-IDRI failed (Retrieval Ratio ~1.0) because the pipeline relied solely on **intensity augmentations** (random windowing). The model simply memorized pixel coordinates or high-level boundaries ("structural bias") without learning semantic features, leading to beautiful but empty attention maps.

The fix was introducing **Strong Spatial Augmentations** (matching standard DINO/ImageNet recipes):
- `RandomResizedCrop(scale=(0.5, 1.0))`
- `RandomHorizontalFlip(p=0.5)`

This forces the model to learn features invariant to zoom and position.
- **Run `nano_aug_test` (2026-02-01):** Achieved **Ratio 3.0** at step 1600 (approx. 100 weight updates), marking the first proof of semantic learning on this dataset.

## Advanced Usage

### Custom Model Configuration

```bash
python scripts/phase5_big_run.py \
    --config custom \
    --vit-patch 16 \
    --vit-dim 768 \
    --vit-depth 12 \
    --vit-heads 12
```

### Override Hardware Detection

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --device cuda \
    --num-workers 4 \
    --pin-memory false
```

### Gradient Accumulation

Achieve large effective batch sizes:

```bash
# Effective batch size = 64 × 8 = 512
python scripts/phase5_big_run.py \
    --config vit-large \
    --batch-size 64 \
    --accumulation-steps 8 \
    --grad-checkpoint
```

### Gradient Checkpointing (Memory Optimization)

Enable gradient checkpointing to save 60-80% of GPU memory:

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --batch-size 128 \
    --grad-checkpoint
```

**How it works:**
- Trades compute for memory by recomputing activations during backward pass
- Essential for large models on 24GB GPUs
- Allows batch_size=64-128 instead of batch_size=16-32
- Adds ~20-30% compute overhead but enables much larger batches

**When to use:**
- ✅ Always use for ViT-Large on 24GB GPUs (4090, 3090, A5000)
- ✅ Recommended for ViT-Giant even on large-memory systems
- ❌ May not be needed for small models or APUs with 128GB+ unified memory

### Mixed Precision Training

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --amp
```

**Note:** Gram Anchoring is **always enabled** and cannot be disabled. It is essential for preventing feature collapse in medical imaging (CT scans). Without it, the model will treat all lung tissue as identical and fail to learn meaningful representations.

## Monitoring and Logging

### Real-Time Metrics

The script logs training progress every 10 seconds:

```
step=   100 loss=3.4521 steps/s=1.23 samples/s=78.7 elapsed=81.3s
step=   200 loss=2.8934 steps/s=1.25 samples/s=80.0 elapsed=160.5s
```

### Visual Monitoring (Heatmaps)

Use the dedicated monitor script to visualize attention heatmaps and check embedding statistics during training:

```bash
# Check the latest checkpoint
python scripts/phase5_monitor.py --checkpoint data/runs/<RUN_ID>/checkpoint_00000100.pth
```

**Outputs:**
- `heatmap.png`: L2 norm of patch tokens (proxy for attention)
- `input.png`: The input slice seen by the model
- `stats.json`: Embedding dispersion statistics (std dev)

**What to look for:**
- **Heatmap:** Should show structure (lung boundaries, vessels, nodules) rather than uniform noise or a single hot spot.
- **Stats:** `std` should be > 0.001. Very low values (e.g., 0.0001) indicate potential feature collapse.

### Checkpoint Frequency

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --ckpt-every 50  # Save every 50 steps (default: 100)
```

### Monitoring Integration (Phase 4)

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --monitor-every 1000  # Generate attention maps every 1000 steps
```

## Anomaly Detection

The script automatically detects and handles training anomalies:

### Loss Spikes
- **Detection**: Loss > 2× recent 10-step average
- **Action**: Log warning, continue training

### Feature Collapse
- **Detection**: Embedding std dev < 0.01
- **Action**: Log critical warning, save emergency checkpoint

### NaN/Inf Loss
- **Detection**: Loss is NaN or Inf
- **Action**: Halt immediately, save emergency checkpoint, raise error

## Reproducibility

Every training run tracks full provenance:

### Git Commit Hash
```json
{
  "git_commit": "a1b2c3d4e5f6-dirty"
}
```

### Data Manifest Hash
```json
{
  "data_manifest_hash": "1234567890abcdef"
}
```

### Configuration Snapshot
All CLI arguments and hardware settings are saved in `data/runs/<run_id>/config.json`.

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or increase accumulation steps:

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --batch-size 32 \
    --accumulation-steps 8
```

### Loss stuck at ~9.01 ("entropy wall")

If `out_dim=8192`, then `ln(out_dim)=9.0109`. A DINO loss that sits near this value for thousands of steps usually means **student and/or teacher outputs are near-uniform**, so the cross-entropy can’t improve.

Try:
- Increase LR (for effective batch 256, try `--lr 5e-4` or `1e-3`).
- Inspect checkpoint center/weights: `python scripts/check_checkpoint.py data/runs/<RUN_ID>/checkpoint_*.pth`.
- Watch TensorBoard entropies (`Train/Entropy_*`) to confirm teacher/student are not saturating.

### Slow Training

Check num_workers and pin_memory settings:

```bash
# Try different num_workers values
python scripts/phase5_big_run.py \
    --config vit-large \
    --num-workers 16
```

### Checkpoint Corruption

If a checkpoint is corrupted, resume from an earlier one:

```bash
# List available checkpoints
ls -lh data/runs/20260103_120000/checkpoint_*.pth

# Resume from earlier checkpoint
python scripts/phase5_big_run.py \
    --config vit-large \
    --resume data/runs/20260103_120000/checkpoint_00000300.pth
```

### Hardware Not Detected

Override device manually:

```bash
python scripts/phase5_big_run.py \
    --config vit-large \
    --device cuda
```

## File Structure

After running Phase 5a, you'll see:

```
data/runs/20260103_120000/
├── config.json                    # Full training configuration
├── checkpoint_00000100.pth        # Checkpoint at step 100
├── checkpoint_00000200.pth        # Checkpoint at step 200
├── checkpoint_00000300.pth        # Checkpoint at step 300
├── checkpoint_00000384.pth        # Final checkpoint (Phase 5a)
└── checkpoint_final_00000384.pth  # Final checkpoint (duplicate)
```

## Next Steps

After Phase 5a completes:
- Proceed to **Phase 6a**: Validate ViT-Large model
- If validation succeeds, proceed to **Phase 5b**: ViT-Giant production run

After Phase 5b completes:
- Proceed to **Phase 6b**: Validate ViT-Giant model and release as production model

## Performance Expectations

### Phase 5a (ViT-Large, 384 steps)
- **Hardware**: RTX 4090 (24GB)
- **Batch size**: 64 (with gradient checkpointing)
- **Accumulation**: 4 steps (effective batch: 256)
- **Duration**: ~2-4 hours
- **Memory usage**: ~9GB (with grad checkpoint) vs ~22GB (without)

### Phase 5b (ViT-Giant, unlimited)
- **Hardware**: AMD Strix Halo (128GB unified)
- **Batch size**: 128 (with gradient checkpointing)
- **Accumulation**: 2 steps (effective batch: 256)
- **Duration**: ~15 days continuous
- **Memory usage**: >24GB (memory wall broken!)

## Example Commands Summary

```bash
# Phase 5a: ViT-Large validation (4090) - WITH gradient checkpointing
python scripts/phase5_big_run.py --config vit-large --max-steps 384 --num-workers 8 --batch-size 64 --accumulation-steps 4 --grad-checkpoint

# Phase 5b: ViT-Giant production (amd395) - WITH gradient checkpointing
python scripts/phase5_big_run.py --config vit-giant --batch-size 128 --accumulation-steps 2 --grad-checkpoint

# Resume from latest checkpoint
python scripts/phase5_big_run.py --config vit-large --resume auto

# Cross-hardware resume (4090 → amd395)
python scripts/phase5_big_run.py --config vit-large --resume data/runs/.../checkpoint_384.pth

# Custom configuration
python scripts/phase5_big_run.py --config custom --vit-patch 16 --vit-dim 768 --vit-depth 12 --vit-heads 12
```

## Support

For issues or questions:
1. Check this documentation first
2. Review logs in `data/runs/<run_id>/`
3. Check anomaly warnings in stdout
4. Verify hardware detection output at script start
5. Consult Phase 3 and Phase 4.5 documentation for baseline behavior

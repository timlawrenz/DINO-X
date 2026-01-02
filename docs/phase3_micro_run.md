# Phase 3: Micro-Run (Fail Fast)

Phase 3 proves the full training code-path works on a tiny scale before starting multi-day training (including HU16 slice loading + random windowing).

## Prereqs
- Phase 1 completed (ROCm + PyTorch functional)
- Phase 2 completed (preprocessed PNGs + index CSV)

Expected Phase 2 index:
- `data/processed/_index/index.csv`

## Run the micro-run

Basic run (writes logs to stdout; checkpoints to `data/runs/...`):

```bash
python3 scripts/phase3_micro_run.py \
  --index-csv data/processed/_index/index.csv \
  --steps 2000 \
  --ckpt-every 200
```

Enable Gram Anchoring:

```bash
python3 scripts/phase3_micro_run.py --gram --gram-weight 1.0
```

Random windowing ranges (only used when Phase 2 data is HU16 grayscale slices):

```bash
python3 scripts/phase3_micro_run.py \
  --rw-level-min -700 --rw-level-max 100 \
  --rw-width-min 300 --rw-width-max 2000
```

## Attention kernel selection (ROCm)

On some ROCm builds, different `scaled_dot_product_attention` backends can change early-run numerics.
If you see suspicious behavior (e.g., loss instantly printing as 0.0), force the reference math backend:

```bash
python3 scripts/phase3_micro_run.py --sdp-backend math
```

If you want to try the experimental AOTriton kernels on AMD (sometimes faster):

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python3 scripts/phase3_micro_run.py --sdp-backend mem_efficient
```

Note: `--sdp-backend` supports `auto`, `math`, and `mem_efficient` (flash may be unavailable on some ROCm builds).


## Overfit Test (Phase 3 exit test)
Run for ~1 hour on the deterministic 1,000-image subset and verify loss drops substantially.

Suggested starting point:

```bash
python3 scripts/phase3_micro_run.py \
  --subset-size 1000 \
  --subset-seed 0 \
  --steps 4000 \
  --batch-size 32
```

## Restart Test (Phase 3 exit test)
1. Start a run, wait for at least one checkpoint.
2. Hit Ctrl+C.
3. Resume from `latest.pth`:

```bash
python3 scripts/phase3_micro_run.py \
  --resume data/runs/phase3_micro_run/<RUN_ID>/checkpoints/latest.pth
```

The loss should continue from approximately where it left off (not restart from scratch).

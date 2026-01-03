# Throughput Tuning Results (2026-01-02 21:13:59)

This note captures the results of the throughput-tuning grid search run stored at:

- `data/runs/throughput_tuning/20260102_211359/`
  - `results.csv`
  - `config.json`
  - `results.json`
  - `subset.json`

## Context

This run is intended to find the **DataLoader throughput knee** (CPU/unified-memory bandwidth saturation point) using **Virtual Inflation** (repeat a 1,000-image subset 1,000× in-memory).

Important: this run used the tuner’s default small model step-loop (not ViT-Giant), so its batch-size ceiling is **not** representative of Phase 5 OOM limits. It *is* representative for DataLoader worker/pinning tuning.

## Dataset + Run Configuration

From `config.json`:

- `index_csv`: `data/processed/_index/index.csv`
- `subset_size`: 1000 (then repeated)
- `inflation_factor`: 1000
- `img_size`: 224
- Random windowing (HU16): `level ∈ [-700, 100]`, `width ∈ [300, 2000]`
- Grid:
  - `batch_size`: `[32, 64, 128, 192, 256, 512]`
  - `num_workers`: `[0, 4, 8, 16, 24, 32]`
  - `pin_memory`: `[True, False]`
- Benchmarking: `warmup_steps=5`, `bench_steps=20`

Environment:
- Torch: `2.9.1+rocm7.1.1.git351ff442`
- HIP: `7.1.52802-26aae437f6`
- Device name: `AMD Radeon Graphics`

## High-Level Outcomes

- Total grid points: **72**
- Successful points (`status=ok`): **50**
- Failures begin at **batch_size=256** when `num_workers>0` (HIP launch failure / illegal address), and all **batch_size=512** points failed.

This failure mode is **not an OOM** signal; it looks like a kernel/runtime error under higher parallelism.

## Key Findings

### 1) `num_workers=0` is severely IO-bound

At `batch_size=32`:
- ~**78 img/s**
- `bound=io_bound`
- `data_ms` dominates (~7.1s over 20 steps)

### 2) The throughput knee is around 8–24 workers

At `batch_size=32`, throughput improves sharply up to ~8 workers and then plateaus:

- `num_workers=4`: ~**337 img/s**
- `num_workers=8`: ~**495 img/s**
- `num_workers=16`: ~**482 img/s**
- `num_workers=24`: ~**503 img/s**
- `num_workers=32`: ~**485 img/s**

Interpretation: **8–24 workers** is the best region; beyond that, extra workers add CPU pressure with minimal gain.

### 3) `pin_memory=True` was consistently better in the high-worker regime

At `batch_size=32`:
- `num_workers=24`: **pin_memory=True ~503 img/s** vs `False ~473 img/s`
- Similar pattern at `num_workers=8/16/32`

### 4) Best-performing configuration in this run

Best row in `results.csv` (highest `images_per_s`):

- `batch_size=32`
- `num_workers=24`
- `pin_memory=True`
- `images_per_s ≈ 502.84`
- `bound=compute_bound`

## Recommendation (for Phase 5 prep)

For future tuning / big-run dry-runs:

- Start from `num_workers=8` or `16` and `pin_memory=True`.
- Only consider `24` if it is consistently better on the *real* model step-loop.
- Treat the `batch_size>=256` failures here as **ROCm/runtime stability** signals (not memory capacity signals).

## Preprocessing Command Used

You ran preprocessing prior to tuning:

```bash
python3 scripts/phase2_preprocess_lidc_idri.py --dicom-root data/raw --out-root data/processed
```

(As of 2026-01-02, the preprocessor defaults to incremental behavior: it skips PNG regeneration for already-existing output series folders and re-indexes existing PNGs into `data/processed/_index/index.csv`.)

# Change: Throughput Tuning (Virtual Inflation)

## Why
Phase 5 (the "Big Run") depends on a stable, high-throughput configuration on Strix Halo.
Because Strix Halo has unusually large unified memory capacity but comparatively lower memory bandwidth, standard batch-size / dataloader rules of thumb are unreliable; we need to experimentally find the "knee of the curve" for throughput before committing to a multi-day run.

## What Changes
- Add a `throughput-tuning` capability to stress-test the training data path and step loop using a limited local image set via "virtual inflation" (repeat the index in memory so the DataLoader behaves like a much larger dataset).
- Add a benchmarking entrypoint `scripts/tune_throughput.py` that grid-searches key throughput parameters (and supports gradient accumulation for effective batch sizing):
  - `batch_size` (find OOM threshold and best stable batch)
  - `num_workers` (find CPU/RAM/unified-memory bandwidth choke point)
  - `pin_memory` (validate Strix Halo APU behavior; sometimes `False` can be faster)
- Emit machine-readable results (CSV/JSON) and a summary recommendation for Phase 5.

## Current Decision (Captured)
- Target Phase 5 validation run: **ViT-Large** (`--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16`).
- Target stability: **effective batch ≥256**.
- Tuned configuration: `--batch-sizes 128 --grad-accum-steps 2 --grad-checkpoint`.
- Observed peak throughput: **~23.5 img/s** on Strix Halo with Triton enabled.

## Impact
- Affected specs: `specs/throughput-tuning/spec.md` (new capability; added requirements).
- Affected code (future implementation): `scripts/tune_throughput.py` plus small shared utilities for timing/memory metrics if needed.
- This change is non-breaking; it adds an interstitial benchmarking step to reduce risk before Phase 5.

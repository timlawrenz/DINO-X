## 1. Implementation
- [x] 1.1 Add `scripts/tune_throughput.py` entrypoint with CLI args for dataset root and output directory.
- [x] 1.2 Implement "virtual inflation" wrapper that repeats an underlying dataset index N times without duplicating files on disk.
- [x] 1.3 Implement grid search over `batch_size`, `num_workers`, and `pin_memory`, including early-stop on OOM.
- [x] 1.4 Add timing breakdown (decode/transform, host-to-device, forward+backward, optimizer step) and throughput metric (images/sec).
- [x] 1.5 Add memory reporting (allocated/reserved/peak if available under ROCm) and basic CPU utilization reporting.
- [x] 1.6 Write results to CSV/JSON and print a "recommended" configuration (knee-of-curve).

## 2. Validation
- [x] 2.1 Run the tuner on the Phase 3 1,000-image subset and verify it completes at least one full grid sweep.
- [ ] 2.2 Confirm outputs are reproducible (same ordering, same grid, stable summary) when rerun with the same seed.
- [x] 2.3 Confirm OOM handling does not crash the runner and correctly records the failure point.

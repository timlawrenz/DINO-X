# Change: Phase 4 Instrumentation (Navigation)

## Why
Phase 5 is a multi-day training run where silent failures (feature collapse, broken preprocessing, regressions after resume) are expensive.
Phase 4 adds lightweight, reproducible instrumentation so we can detect problems early without stopping training.

## What Changes
- Define a `training-monitor` capability that can be executed periodically (e.g., every 24h) alongside training to:
  - Generate an attention map visualization for a fixed, known test sample.
  - Compute an embedding-dispersion metric (standard deviation) on a fixed batch to detect collapse.
- Define a `dataset-split` capability that reserves a deterministic 10% validation set (no leakage) before Phase 5 begins.

## Impact
- Affected specs: `specs/training-monitor/spec.md`, `specs/dataset-split/spec.md` (new capabilities; added requirements).
- Affected code (future implementation): a Phase 4 monitoring script under `scripts/`, dataset split manifest output under `data/processed/`, and small updates to training entrypoints to honor the reserved validation split.
- This change is non-breaking; it adds Phase 4 navigation tools required to safely execute Phase 5.
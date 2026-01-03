# Change: Phase 4 Instrumentation (Navigation)

## Why
Phase 5 is a multi-day training run where silent failures (feature collapse, broken preprocessing, regressions after resume) are expensive.
Phase 4 adds lightweight, reproducible instrumentation so we can detect problems early without stopping training.

## What Changes

- **TensorBoard Integration:** Integrate `torch.utils.tensorboard` into the main training loop (`scripts/phase5_big_run.py`) to log real-time metrics:

  - Scalars: Loss, Learning Rate, Throughput (samples/s), Embedding Std Dev.

  - Images: Periodic Attention Heatmaps (visualizing feature learning).

- Define a `dataset-split` capability that reserves a deterministic 10% validation set (no leakage) before Phase 5 begins.

- Maintain `scripts/phase5_monitor.py` as a standalone inspection tool for deep-dives on specific checkpoints.



## Impact

- Affected specs: `specs/training-monitor/spec.md`, `specs/dataset-split/spec.md` (new capabilities; added requirements).

- Affected code:

  - Update `scripts/phase5_big_run.py` to initialize SummaryWriter and log events.

  - Update `requirements.txt` to include `tensorboard`.

- This change is non-breaking; it adds Phase 4 navigation tools required to safely execute Phase 5.

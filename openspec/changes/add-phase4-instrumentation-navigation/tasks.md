## 1. Implementation
- [x] 1.1 Add a deterministic dataset split generator that reserves 10% validation data (no leakage) and writes a split manifest.
- [x] 1.2 Update training entrypoints / dataloaders to accept a split manifest and exclude validation data from training.
- [x] 1.3 Add a Phase 4 monitor entrypoint that loads a checkpoint/run and produces:
  - attention map visualization for a fixed test sample
  - embedding dispersion metric(s) over a fixed batch
- [x] 1.4 Ensure monitor outputs are written to a separate, timestamped output directory and are safe to produce while training is running.
- [x] 1.5 Document how to create the validation split and how to run the monitor manually (and via cron/systemd).

## 2. Validation
- [x] 2.1 Generate a split manifest and verify train/val are disjoint and counts match expectations.
- [x] 2.2 Run the monitor against a known checkpoint and confirm it emits an attention visualization artifact and a metrics JSON.
- [x] 2.3 Run the monitor twice on the same checkpoint/sample and confirm outputs are stable (within expected numeric tolerance).
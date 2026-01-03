## 1. Implementation
- [x] 1.1 Add a deterministic dataset split generator that reserves 10% validation data (no leakage) and writes a split manifest.
- [x] 1.2 Update training entrypoints / dataloaders to accept a split manifest and exclude validation data from training.
- [x] 1.3 Add a Phase 4 monitor entrypoint that loads a checkpoint/run and produces attention maps and metrics.
- [ ] 1.4 **TensorBoard Integration:**
  - Add `tensorboard` to `requirements.txt`.
  - Update `scripts/phase5_big_run.py` to initialize `SummaryWriter`.
  - Log scalar metrics (Loss, LR, Steps/s, Samples/s) every `monitor_every` steps (or more frequently).
  - Log attention heatmaps and input slices as images every `monitor_every` steps.
  - Log embedding std dev stats every `monitor_every` steps.

## 2. Validation
- [x] 2.1 Generate a split manifest and verify train/val are disjoint and counts match expectations.
- [x] 2.2 Run the monitor against a known checkpoint and confirm it emits an attention visualization artifact and a metrics JSON.
- [ ] 2.3 Verify TensorBoard logs appear in `data/runs/<run_id>` and can be viewed via `tensorboard --logdir data/runs`.
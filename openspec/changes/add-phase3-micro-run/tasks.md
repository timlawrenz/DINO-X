## 1. Implementation
- [x] 1.1 Add a Phase 3 training entrypoint (micro-run) that runs on Strix Halo ROCm.
- [x] 1.2 Implement/port the DINOv3 student-teacher loop with Gram Anchoring enabled and configurable.
- [x] 1.3 Implement deterministic dataset subsetting to exactly 1,000 preprocessed images (seeded), with an explicit "overfit" mode.
- [x] 1.4 Implement checkpoint save/load for model, optimizer, scaler (if used), RNG state, and training step.
- [x] 1.5 Add safe interrupt handling (Ctrl+C) that triggers a final checkpoint write and clean shutdown.
- [x] 1.6 Add the Overfit Test command + expected behavior documentation (loss drops near zero within ~1 hour).
- [x] 1.7 Add the Restart Test command + verification instructions (resume continues loss curve).

## 2. Validation
- [x] 2.1 Run micro-run overfit on 1,000 images for ~1 hour and save logs (loss curve).
- [x] 2.2 Interrupt and resume from the last checkpoint; confirm loss continues from the prior value.

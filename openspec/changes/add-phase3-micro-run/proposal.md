# Change: Phase 3 Micro-Run (Fail Fast)

## Why
Phase 3 is the first time DINO-X runs an end-to-end training loop (data 0 model 0 loss 0 optimizer) on Strix Halo.
Before starting multi-day training, we must prove the training code path is correct, stable under ROCm, and resumable via checkpoints.

## What Changes
- Define a `micro-run` capability that implements a minimal DINOv3 training loop with **Gram Anchoring** enabled.
- Add robust checkpoint save/resume (including safe interrupt handling) so long-running training can recover from failures.
- Add a deterministic "micro dataset" mode (1,000 images) and two formal exit tests:
  - the **Overfit Test** (loss drops near zero in ~1 hour)
  - the **Restart Test** (Ctrl+C + resume continues from the prior loss)

## Impact
- Affected specs: `specs/micro-run/spec.md` (new capability, added requirements).
- Affected code (future implementation): new training entrypoint under `scripts/` and/or `src/`, checkpoint outputs under `data/` (ignored), and minimal docs describing how to run the micro-run.
- This change is non-breaking; it creates the Phase 3 training foundation required for Phase 4 instrumentation and Phase 5 full training.

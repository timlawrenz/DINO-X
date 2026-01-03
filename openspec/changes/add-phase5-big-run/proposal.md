# Change: Phase 5 Big Run (Dual Execution Strategy)

## Why
Phase 5 represents the culmination of the DINO-X project: training a high-fidelity Vision Foundation Model on medical CT data.
We need a parameterized training script that can execute both a validation run (ViT-Large on 4090) and a production run (ViT-Giant on amd395) without code duplication.
The validation run (Phase 5a) proves convergence within budget constraints, while the production run (Phase 5b) breaks the 24GB memory wall.

## What Changes
- Define a `big-run` capability that implements a production-grade DINOv3 training script with full model scaling support.
- Create a single parameterized training script that accepts model configuration (ViT-Large vs ViT-Giant) and hardware target (4090 vs amd395) as CLI arguments.
- Support two distinct execution modes:
  - **Phase 5a (ViT-Large)**: 384-step validation run on RTX 4090 with specific hyperparameters (`--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16 --num-workers 8`)
  - **Phase 5b (ViT-Giant)**: Full 15-day production run on AMD Strix Halo (amd395) with memory-breaking configuration
- Extend checkpoint system from Phase 3 to handle long-running training with automatic resumption.
- Add configuration presets for both ViT-Large and ViT-Giant architectures to avoid parameter duplication.
- **Technical Fixes**: Ensure robust implementation of DINOv3 objectives by including:
  - Multi-view augmentation (2 differently windowed views per sample).
  - Teacher centering and sharpening (`DINOLoss`) to prevent feature collapse.
  - Volumetric 3-slice input contexts ($z-1, z, z+1$).

## Impact
- Affected specs: `specs/big-run/spec.md` (new capability, added requirements).
- Affected code (future implementation):
  - New parameterized training script under `scripts/phase5_big_run.py`
  - Model configuration system (likely YAML or CLI presets)
  - Extended checkpoint management for multi-day runs
  - Hardware-specific optimization paths (ROCm vs CUDA)
- Dependencies: Builds on Phase 3 (micro-run checkpointing), Phase 4 (monitoring), and Phase 4.5 (throughput tuning).
- This change is non-breaking; it creates the production training infrastructure for both validation and production runs, with corresponding Phase 6 validation workflows.

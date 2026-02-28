# DINO-X Copilot Instructions

Project DINO-X is an experimental research initiative to train a high-fidelity Vision Foundation Model (VFM) for volumetric medical imaging (chest CT) using consumer-grade hardware—specifically AMD Strix Halo with ROCm 7.1.

## Environment Setup

### Hardware Requirements
- **Primary trainer**: AMD Ryzen AI MAX+ 395 (Strix Halo) with 128GB unified memory (~96GB usable as VRAM)
- **Optional worker**: NVIDIA RTX 4090 (24GB) for preprocessing and validation

### Software Stack
- **OS**: Linux (kernel 6.11+; 6.15+ preferred for gfx1151/ROCm support)
- **Compute**: ROCm 7.1 with gfx1151 support
- **Python**: 3.13+ (in `.venv`)
- **PyTorch**: ROCm 7.1 build with Flash Attention 2 via Triton backend

### ROCm Environment
Before running any scripts, source the ROCm environment helper:

```bash
source scripts/rocm_env.sh         # bash/zsh
source scripts/rocm_env.fish       # fish shell
```

This sets `PATH` and `LD_LIBRARY_PATH` to include `/opt/rocm/bin` and `/opt/rocm/lib`.

### Python Environment
Install dependencies into the virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
./setup_rocm_8060s.sh              # Install ROCm PyTorch wheels
pip install -r requirements.txt
```

Dependencies are tracked in:
- `requirements.in` - Minimal human-maintained dependencies
- `requirements.txt` - Fully pinned snapshot from `.venv` (auto-generated)

## Running Scripts

### Validation & Testing
```bash
# Phase 1: Validate attention kernels (smoke test)
python scripts/phase1_validate_attention.py --size 512 --device cuda --dtype fp16

# Phase 2: Validate preprocessed data samples
python scripts/phase2_validate_samples.py

# View Retrieval Eval (label-free representation check)
python scripts/phase5_view_retrieval_eval.py --checkpoint path/to/checkpoint.pth
```

### Training Phases
```bash
# Phase 3: Micro-run (overfit test on 1,000 images)
python scripts/phase3_micro_run.py --data-dir data/processed --steps 100

# Phase 5: Big Run (full training)
python scripts/phase5_big_run.py \
  --data-dir data/processed \
  --checkpoint-dir checkpoints/ \
  --vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16 \
  --batch-size 128 --grad-accum-steps 2 --grad-checkpoint
```

### Monitoring & Utilities
```bash
# Monitor training progress
python scripts/phase5_monitor.py --checkpoint-dir checkpoints/

# Check checkpoint contents
python scripts/check_checkpoint.py checkpoints/step_1000.pth

# Throughput tuning (grid search batch size & num_workers)
python scripts/tune_throughput.py --batch-sizes 32,64,128 --num-workers 0,8,16
```

## Architecture Overview

### Data Pipeline (2.5D Volumetric Slices)
1. **Raw Format**: LIDC-IDRI DICOM volumes converted to 16-bit PNG slices (one file per slice)
   - Stores full Hounsfield Unit (HU) range: [-1000, 4000] offset to uint16
   - See `scripts/phase2_preprocess_lidc_idri.py`

2. **Training Loader**: `PngDataset` in training scripts
   - Loads 3 consecutive slices (z-1, z, z+1) and stacks as RGB-like (3, H, W) tensor
   - Applies **random windowing** on-the-fly: randomly samples Level ∈ [-400, 400] and Width ∈ [800, 2000] to simulate radiologist viewing modes
   - Spatial augmentations: RandomResizedCrop (scale 0.5-1.0) + RandomHorizontalFlip
   - Normalizes with ImageNet statistics (mean: 0.485, 0.229, 0.456; std: 0.229, 0.224, 0.225)

3. **Index File**: `data/processed/_index/index.csv` or split manifests (`data/processed/_index/train.csv`, `val.csv`)
   - CSV format: `SeriesInstanceUID,SliceIndex,FilePath`

### Model: DINOv3 + Gram Anchoring
- **Backbone**: Vision Transformer (ViT-Large or ViT-Giant) with patch-based attention
- **Training**: Self-supervised learning via student-teacher distillation (EMA teacher)
- **Loss Components**:
  - Cross-entropy distillation loss between student/teacher outputs
  - **Gram Anchoring**: Texture regularizer that preserves geometric correlations between patches (prevents feature collapse in medical images)
- **Architecture Configs**:
  - ViT-Large: `--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16`
  - ViT-Giant: Memory-breaking config (>24GB VRAM; designed for Strix Halo)

### Checkpointing & Resume
- Checkpoints saved as `.pth` files every N steps
- Contains: `step`, `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `rng_state`
- Safe Ctrl+C handling: training scripts catch SIGINT and save checkpoint before exit
- Resume with `--resume path/to/checkpoint.pth`

### Monitoring & Evaluation
- **Attention Maps**: Visualize patch token L2 norms as heatmaps (should highlight nodules)
- **Embedding StdDev**: Track standard deviation across patient embeddings (detect collapse if σ → 0)
- **View Retrieval Eval** (Phase 5): Label-free representation quality check
  - Two augmented views of same image should retrieve each other (nearest neighbor)
  - Success: Retrieval ratio > 3.0x random baseline
- **Linear Probe** (Phase 6): Freeze backbone, train logistic regression on malignancy labels
  - Target: AUC > 0.90

## Key Conventions

### Code Style
- Python follows PEP 8 with type hints where practical
- Favor explicit, readable code over heavy abstraction
- Keep modules focused (single-purpose files in `src/`)
- Configuration via YAML in `configs/` (minimal hard-coded hyperparameters)

### Data Handling
- **Never commit raw medical data or large checkpoints** to Git
- Store data under `data/` (gitignored)
- Store checkpoints under `checkpoints/` or `runs/` (gitignored)
- Include provenance manifests (CSV/JSON) for reproducibility

### Testing Strategy
- **Smoke tests**: Short validation scripts that catch regressions (Phase 1, Phase 2 validators)
- **Overfit tests**: Micro-runs on 1,000 images to verify gradient path (Phase 3)
- **Long runs**: Multi-day experiments with monitoring hooks (Phase 5)
- No formal unit test framework; validation is scenario-based via phase scripts

### Git Workflow
- Simple feature-branch workflow (`main` stays stable)
- Descriptive commit messages referencing experiments (e.g., `exp: tune gram-anchoring coef`, `feat: add 2.5d slice loader`)
- Large experimental checkpoints not committed

## Documentation Structure

- **`README.md`**: High-level project overview and mission
- **`docs/roadmap.md`**: Phase 1-6 execution plan with success criteria
- **`docs/hardware_setup.md`**: Platform bootstrap (ROCm installation, Flash Attention setup)
- **`docs/data_preprocessing.md`**: Detailed LIDC-IDRI preprocessing pipeline (DICOM → PNG → tensor)
- **`docs/EXPERIMENTS.md`**: Experiment logs and hyperparameter tuning results
- **`openspec/project.md`**: Project conventions, tech stack, constraints
- **`openspec/AGENTS.md`**: OpenSpec workflow for AI assistants (spec-driven development)
- **`openspec/changes/`**: Active change proposals (what SHOULD change)
- **`openspec/specs/`**: Current specifications (what IS built)

### Creating New Documentation
- General documentation goes in `docs/`
- Specifications and change proposals follow OpenSpec structure in `openspec/`
- **Do not create markdown files for planning, notes, or tracking** outside these locations

## OpenSpec Workflow

This project uses OpenSpec for spec-driven development. When making significant changes:

1. **Check existing specs**: `openspec list` and `openspec list --specs`
2. **Read project context**: `openspec/project.md` and `openspec/AGENTS.md`
3. **Create change proposal**: Follow the workflow in `openspec/AGENTS.md`
   - Use verb-led kebab-case IDs (e.g., `add-phase6-validation`)
   - Include `proposal.md`, `tasks.md`, and spec deltas
   - Validate with `openspec validate <change-id> --strict`
4. **Implement after approval**: Complete tasks in `tasks.md` sequentially
5. **Archive after deployment**: `openspec archive <change-id> --yes`

### When to Skip OpenSpec
- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Configuration changes
- Tests for existing behavior

## Domain Context

### Medical Imaging Specifics
- **Input**: Chest CT scans from LIDC-IDRI dataset (512×512 axial slices, ~240 per volume)
- **HU Values**: Hounsfield Units measure tissue density (-1000=air, 0=water, +400-1000=bone)
- **Windowing**: Radiologists adjust display ranges to view different tissue types (lung vs soft tissue vs bone)
- **2.5D Context**: Use 3 consecutive slices to provide volumetric context while keeping input size manageable

### Why Standard Vision Models Fail
- General-purpose models (DINOv2 on ImageNet) rely on edges and colors
- Medical images rely on subtle texture differences in grayscale
- Solution: DINOv3 with Gram Anchoring + random windowing augmentation

### Reproducibility Requirements
- All experiments re-runnable from config file + seed
- Infrastructure as code (setup scripts, validation scripts)
- Clear documentation for third-party verification
- Provenance manifests for data and checkpoints

## Critical Constraints

### Hardware Limitations
- Single-device training (no distributed training)
- Limited memory bandwidth (~256 GB/s) → 14-21 day training runs
- Requires robust desktop cooling (sustain ~120W TDP for weeks)
- Laptop chassis considered non-viable for full runs

### Software Requirements
- ROCm 7.1 with gfx1151 support is mandatory
- PyTorch must use hipBLASLt and Flash Attention 2 (Triton backend)
- Set `export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for mem-efficient SDPA

### Stability First
- Multi-day training runs: failures late in training are costly
- No NaNs, no hangs, no OOM crashes
- Checkpointing every N steps with safe interrupt handling
- Monitor thermal logs and validation outputs daily

### Privacy Constraints
- Medical data must remain local (HIPAA/GDPR compliance)
- Data never leaves secure infrastructure
- No uploading raw volumes or patient identifiers

# Project Context

## Purpose
Project DINO-X is an experimental research initiative to train a **high-fidelity Vision Foundation Model (VFM)** for volumetric medical imaging (chest CT) using **consumer-grade hardware**, specifically AMD Strix Halo with ROCm 7.1. The primary goal is to produce an open-weight ViT-Giant (\~1B+ parameters) backbone that learns robust, texture-aware representations of lung tissue from unlabeled CT volumes, enabling downstream cancer detection and segmentation with dramatically fewer labeled examples. The project is also a hardware and systems validation of the "high-capacity, high-latency" paradigm: proving that long-horizon, data-center-class training runs are feasible on a single, well-cooled Strix Halo workstation.

## Tech Stack
- **Hardware**
  - Trainer: AMD Ryzen AI MAX+ 395 (Strix Halo) with 128GB Unified Memory (up to \~96GB usable as VRAM).
  - Optional worker: NVIDIA RTX 4090 (24GB) for preprocessing, validation experiments, and baselines.
- **Operating System & Drivers**
  - Linux (kernel 6.11+; 6.15+ preferred for optimal gfx1151/ROCm support and unified memory stability).
  - ROCm 7.1 with official **gfx1151** support.
- **Core Libraries**
  - Python 3.x.
  - PyTorch built with ROCm 7.1 (Flash Attention 2 + Triton enabled, hipBLASLt enabled for GEMM).
  - Flash Attention 2 (via Triton or Composable Kernel backends) for ViT attention.
- **Model & Training**
  - DINOv3-style Vision Transformer (ViT-Giant or larger) with **Gram Anchoring**.
  - 2.5D CT input (3-slice stacks) mapped into standard RGB channels.
  - YAML-based experiment configs under `configs/` and Python training code under `src/`.
- **Data & IO**
  - LIDC-IDRI chest CT dataset (DICOM) converted to normalized 2D slices.
  - Custom data pipeline for random windowing and volumetric slicing (see `docs/data_pipeline.md`).

## Project Conventions

### Code Style
- Python code follows **PEP 8** with type hints where practical; favor explicit, readable code over heavy abstraction.
- Keep modules focused and relatively small (aim for single-purpose files in `src/`); prefer pure functions for data transforms and thin orchestration in training scripts.
- Configuration is **declarative** via YAML in `configs/`, with minimal hard-coded hyperparameters; document non-obvious behaviors in docstrings or short comments.

### Architecture Patterns
- Clear separation of concerns:
  - **Data pipeline**: DICOM loading, windowing, 2.5D slice construction, and augmentations.
  - **Model**: ViT/DINOv3 backbone with Gram Anchoring loss components and attention configuration.
  - **Training loop**: optimization, checkpointing, logging, and resume logic.
  - **Evaluation**: linear probes, segmentation heads, and attention-map visualization (notebooks + scripts).
- Single-device training on Strix Halo with no distributed training; scale comes from maximizing batch size via unified memory, Flash Attention 2, hipBLASLt, and gradient checkpointing.
- Reproducibility is a goal: experiments should be re-runnable from a config file + seed, with all changes tracked via Git.

### Testing Strategy
- **Unit tests / smoke tests** for:
  - Data transforms and windowing (including 2.5D slice creation) on small synthetic volumes.
  - Model construction and forward passes on tiny inputs to catch ROCm/kernel regressions.
- **Integration / system tests**:
  - Short "smoke" training runs (\<1 epoch on a small subset of LIDC-IDRI) to validate that a full pipeline (data → model → optimizer → checkpoint) executes under ROCm 7.1 without errors.
  - Periodic validation scripts / notebooks to confirm linear-probe AUC and segmentation Dice on small held-out subsets.
- Long multi-day training runs are treated as experiments; add monitoring and logging hooks (loss curves, GPU utilization, VRAM usage) to quickly diagnose instability.

### Git Workflow
- Use a simple **feature-branch** workflow:
  - `main` (or `master`) remains stable and runnable.
  - Short-lived branches for experiments and features (e.g., `feat/random-windowing`, `exp/vit-giant-512px`).
- Prefer descriptive commit messages that reference the experiment or capability being changed (e.g., `exp: tune gram-anchoring coef`, `feat: add 2.5d slice loader`).
- Large experimental checkpoints and raw data are **not** committed; store only configs, code, and lightweight logs.

## Domain Context
- Domain: **Chest CT medical imaging** with emphasis on lung cancer detection using the **LIDC-IDRI** dataset.
- Input data consists of 3D CT volumes (512×512 axial slices, \~240 slices per scan on average), with severe class imbalance (few nodules among many normal slices) and high information density (12–16-bit HU values).
- Standard general-purpose vision models underperform here due to distribution shift; DINO-X applies **self-supervised DINOv3 with Gram Anchoring** and **random windowing** of CT Hounsfield Unit ranges to learn texture-sensitive, anatomy-aware representations suitable for dense prediction tasks (segmentation, nodule localization).
- The project explicitly targets the "garage-scale" research setting: enabling high-end medical foundation models to be trained on-premise without data ever leaving hospital or lab infrastructure.

## Important Constraints
- **Hardware constraints**:
  - Training is designed for a **single Strix Halo** device with \~96GB VRAM-equivalent unified memory and limited memory bandwidth (~256 GB/s), leading to training runs on the order of **14–21 days**.
  - Requires robust, desktop-class cooling to sustain \~120W TDP for weeks; thin-and-light laptop chassis are considered non-viable for full runs.
- **Software constraints**:
  - ROCm 7.1 with **gfx1151** support is mandatory; PyTorch must be built/configured to use **hipBLASLt** and **Flash Attention 2** (Triton backend) for acceptable throughput.
  - Linux kernel 6.11+ is required, with newer kernels (\>=6.15) preferred for stable unified memory and NPU/GPU management.
- **Experiment constraints**:
  - Training assumes tolerance for long, uninterrupted runs; failures late in training are costly, so stability (no hangs, no NaNs) is a first-class requirement.
  - Batch sizes should exploit large VRAM (64–128 samples at 512×512 when feasible) to realize the scientific goals.
- **Data & privacy constraints**:
  - Medical data (LIDC-IDRI or hospital data) must be handled under standard privacy regulations (e.g., HIPAA/GDPR); data is expected to remain local to secure infrastructure.

## External Dependencies
- **Hardware / Platform**
  - AMD Strix Halo platform (Ryzen AI MAX+ 395) with LPDDR5x-8000 and unified memory; optionally, an RTX 4090 workstation for auxiliary workloads.
- **Software stack**
  - AMD ROCm 7.1 drivers and libraries (rocBLAS, **hipBLASLt**, MIOpen, etc.).
  - PyTorch with ROCm backend, compiled/installed with Flash Attention 2 and Triton for gfx1151.
  - Linux kernel 6.11+ with appropriate amdgpu firmware.
- **Data sources**
  - LIDC-IDRI dataset from The Cancer Imaging Archive (TCIA) as the primary training corpus.
- **Tooling & ecosystem**
  - OpenAI Triton or AMD Composable Kernel for optimized attention kernels.
  - Jupyter/Notebook tooling for analysis and visualization (probing embeddings, attention maps, and evaluation metrics).

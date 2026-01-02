# Project DINO-X: Self-Supervised Medical Vision Backbone

Project DINO-X is an experimental research initiative to train a high-fidelity Vision Foundation Model (VFM) for medical imaging (CT Volumetric Data) using consumer-grade hardware.

By leveraging the massive Unified Memory Architecture of the AMD Strix Halo, this project aims to break the "Memory Wall" that currently restricts the training of Billion-Parameter models to enterprise data centers.

## 🎯 The Mission
General-purpose vision models (like DINOv2 trained on ImageNet) excel at recognizing objects defined by edges and colors (cars, dogs, faces). However, they frequently fail in medical imaging, which relies on subtle texture differences (tissue density, nodules vs. vessels) in grayscale environments.

**The Goal:** Train a ViT-Giant (1B+ params) backbone purely on unlabeled Chest CT volumes that understands lung tissue structure better than any supervised model currently available.

**The Output:** A pre-trained `.pth` backbone released open-source, allowing researchers to train cancer detection classifiers with 95% fewer labeled images.

## 🧬 The Core Hypothesis

### 1. The Hardware Strategy: "High-Capacity, High-Latency"
Standard SSL (Self-Supervised Learning) requires massive batch sizes (256+) to stabilize training. This typically requires an NVIDIA H100 (80GB).

- **The Constraint:** A consumer RTX 4090 (24GB) cannot fit the batch size required for a ViT-Giant.
- **The Solution:** We utilize the AMD Strix Halo (128GB Unified Memory).
- **The Trade-off:** We accept slower training times (lower bandwidth) in exchange for the sheer memory capacity required to fit the model and batch size.

### 2. The Algorithmic Strategy: Gram Anchoring
We utilize DINOv3 (August 2025), specifically enabling Gram Anchoring.

- **Problem:** Standard models suffer from "feature collapse" in medical images, treating all lung tissue as identical.
- **Solution:** Gram Anchoring forces the model to respect the geometric texture correlations between patches, preserving the fine details of nodules and ground-glass opacities.

## 🛠️ Tech Stack Overview

**Hardware:**
- **Trainer:** AMD Strix Halo (128GB Unified RAM) - Allocated as 96GB VRAM.
- **Worker:** NVIDIA RTX 4090 (24GB) - Preprocessing & Validation.

**Software:**
- **OS:** Linux Kernel 6.11+
- **Compute:** ROCm 7.1 (GFX1151 support)
- **Optimization:** Flash Attention 2 (via Triton/CK backend)

**Data:**
- **Source:** LIDC-IDRI (The Cancer Imaging Archive).
- **Format:** 2.5D Volumetric Slices (3-channel windowing).

## 📊 Criteria for Success

We define success not just by "the code runs," but by three scientific benchmarks:

1. **Linear Probe Accuracy:** A simple logistic regression trained on top of the frozen backbone must achieve > 0.90 AUC on nodule malignancy detection.

2. **No Feature Collapse:** Embeddings for different patient scans must remain distinct (verified via standard deviation analysis).

3. **Attention Map Fidelity:** The model must be able to "segment" a lung nodule via its self-attention maps without ever being trained on segmentation labels.

## 📂 Repository Structure

Key documentation and Phase 1 tooling live in `docs/` and `scripts/`.

```
/
├── docs/
│   ├── roadmap.md                     # Project roadmap (Phase 1–6)
│   ├── hardware_setup.md              # Phase 1 platform bootstrap guide
│   └── ROCm 7.1 DINO-X Viability Review.md
├── openspec/
│   ├── project.md                     # Project conventions/context
│   └── changes/                       # OpenSpec proposals and task checklists
├── scripts/
│   ├── phase1_validate_attention.py   # Phase 1 attention smoke test
│   ├── rocm_env.sh                    # ROCm env helper (bash/zsh)
│   └── rocm_env.fish                  # ROCm env helper (fish)
├── setup_rocm_8060s.sh                # Install ROCm PyTorch wheels into .venv
├── requirements.in                    # Minimal human-maintained deps
├── requirements.txt                   # Fully pinned snapshot (generated from .venv)
└── README.md
```

## 📜 Citation & License
This project is currently in the **Planning / Pre-Production** phase.

- **License:** GPLv3 (see `LICENSE`)
- **Base Architecture:** Meta Research / DINOv3

> **Note:** This project pushes consumer hardware to its absolute thermal and memory limits. It is designed as a "Stress Test" for the concept of Garage-Scale Foundation Models.

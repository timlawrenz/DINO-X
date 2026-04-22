# DINO-X: Open-Source Medical Imaging Model Zoo

DINO-X is an open-source model zoo for medical imaging, training **pan-organ,
modality-specific** Vision Foundation Models using self-supervised learning. Each
model is **scale-aware**: it natively understands the physical dimensions of what
it sees, not just the pixel grid.

**Planned models:**

| Model | Modality | Input | Status |
|-------|----------|-------|--------|
| `dinox-ct-vit-small` | CT (all organs) | 2.5D slices + spacing | MVP proven |
| `dinox-ct-vit-large` | CT (all organs) | 2.5D slices + spacing | Planned |
| `dinox-mri-vit-small` | MRI (all organs) | 2.5D slices + spacing | Planned |
| `dinox-xray-vit-small` | X-ray (all views) | 2D + pixel spacing | Planned |

## Why DINO-X?

General-purpose vision models (DINOv2, ImageNet backbones) fail on medical images
because they rely on edges and colors. Medical imaging relies on **subtle texture
differences** in grayscale — tissue density, nodule morphology, ground-glass
opacities.

Worse, existing medical foundation models are trained on narrow, single-organ
datasets. [Recent analysis](https://huggingface.co/papers/2603.27460) catalogs
over 1,000 medical datasets and finds the data trapped in fragmented silos.
DINO-X consolidates these silos into coherent, pan-organ pretraining.

### Scale Awareness

Vision Transformers process images in fixed pixel grids (e.g., 14×14 patches).
But in medical imaging, **physical size is everything**:

- A 14×14 patch at 0.5mm spacing covers **7×7mm** of tissue
- The same patch at 1.5mm spacing covers **21×21mm** of tissue

A 7mm lung nodule warrants follow-up; a 21mm nodule triggers immediate biopsy.
Standard ViTs are blind to this distinction.

DINO-X solves this with a **ScaleEmbedding**: the three physical dimensions from
the DICOM header — `pixel_spacing_x`, `pixel_spacing_y`, `slice_thickness` — are
passed through a lightweight MLP and added to the patch embeddings. The model
learns natively that `[0.5, 0.5, 1.0]` represents a tight microscopic area while
`[1.5, 1.5, 5.0]` represents a macroscopic overview. The spacing is projected as
a **continuous value** (not a categorical bucket) so the model generalizes across
the full range of clinical scanners.

### Proven Results

The MVP two-organ ablation (2026-04-21) on 43K slices from LIDC-IDRI (lung) +
Pancreas-CT (abdomen) demonstrates the scale embedding's impact:

| Arm | Final Loss | Steps | Note |
|-----|-----------|-------|------|
| Baseline (no scale) | 8.992 | 5,000 | Stuck at entropy wall |
| **Scale-aware** | **0.134** | 5,000 | Breaks through decisively |

Both models are healthy (no feature collapse). Full results in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Quick Start

### For Researchers (using a pretrained model)

```python
# Planned API (dinox-hub package)
from dinox_hub import load_model

model = load_model("dinox-ct-vit-small")

# Pass raw pixels + DICOM spacing — no resampling needed
import numpy as np
pixels = np.load("my_ct_slice.npy")          # raw HU array from PACS
spacing = (0.703, 0.703, 1.25)               # from DICOM header
features = model.encode(pixels, spacing)      # → (1, dim) embedding

# Fine-tune with LoRA (5MB adapter, frozen backbone)
from dinox_hub import apply_lora
model = apply_lora(model, rank=8)             # only adapter trains
```

### For Contributors (training from source)

```bash
git clone https://github.com/timlawrenz/DINO-X.git && cd DINO-X
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch preprocessed data from HuggingFace
export HF_TOKEN=hf_...
bash scripts/fetch_hf_data.sh

# Train baseline ViT-Small (5K steps, ~30 min on RTX 3090)
python scripts/phase5_big_run.py \
  --config vit-small --max-steps 5000 \
  --batch-size 64 --accumulation-steps 4 \
  --index-csv data/processed/combined-mvp/index.csv

# Train with scale awareness
python scripts/phase5_big_run.py \
  --config vit-small --max-steps 5000 \
  --batch-size 64 --accumulation-steps 4 \
  --scale-aware \
  --index-csv data/processed/combined-mvp/index.csv
```

## Architecture

### Training: DINOv3 + Gram Anchoring + ScaleEmbedding

- **Self-supervised**: Student-teacher distillation with EMA (no labels needed)
- **Gram Anchoring**: Texture regularizer that preserves geometric correlations
  between patches (prevents feature collapse on medical images)
- **ScaleEmbedding**: Continuous physical-scale injection from DICOM metadata
- **Random Windowing**: On-the-fly HU window augmentation simulating different
  radiologist viewing modes (lung, soft tissue, bone)

### Data Pipeline

```
DICOM volumes → 16-bit HU PNG slices → 2.5D stacking (3 consecutive slices)
                                      → Random windowing augmentation
                                      → ScaleEmbedding from DICOM spacing
```

**Current datasets** (hosted on [HuggingFace](https://huggingface.co/datasets/timlawrenz/dinox-mvp-data)):

| Dataset | Organ | Slices | Source |
|---------|-------|--------|--------|
| LIDC-IDRI | Lung | 24,441 | [TCIA](https://www.cancerimagingarchive.net/) |
| Pancreas-CT | Abdomen | 18,942 | [TCIA](https://www.cancerimagingarchive.net/) |

### Data Lineage

Every training run tracks full provenance via the `zoo/` package:

- **DatasetRegistry** (`zoo/registry.py`): YAML-backed catalog of all datasets
- **DataManifest** (`zoo/manifest.py`): Parquet per-slice metadata
- **DataLineage** (`zoo/lineage.py`): Git commit, config hash, data hash per run
- **DatasetMerger** (`zoo/merge.py`): Weighted multi-dataset sampling

### Downstream Adaptation (LoRA / PEFT)

DINO-X is designed for parameter-efficient fine-tuning. Researchers download
the frozen backbone and train a tiny adapter (~5MB) for their specific task:

```python
from zoo.hub import load_model
from zoo.peft import apply_lora, save_adapter

# Load pretrained backbone
model = load_model("timlawrenz/dinox-ct-vit-small-v1")

# Inject LoRA — only adapter weights train (~200K params)
model = apply_lora(model, rank=8)

# ... train on your labeled data ...

save_adapter(model, "my-pe-adapter/")  # ~0.8 MB
```

Or use the fine-tuning script directly:

```bash
python scripts/finetune_lora.py \
  --backbone timlawrenz/dinox-ct-vit-small-v1 \
  --train-csv data/train_labels.csv \
  --val-csv data/val_labels.csv \
  --task classification --num-classes 5 \
  --rank 8 --epochs 50 --lr 1e-3 \
  --output adapters/my-adapter/
```

Key design decisions:
- **HF peft integration**: LoRA injection targeting `qkv`, `proj`, `fc1`, `fc2`
- **ScaleEmbedding always frozen**: Adapters learn pathology, not alternate physics
- **Head outside PEFT**: Clean save/load — adapter weights via peft, task head separately

## Repository Structure

```
├── configs/
│   └── panorgan_ct_vits.yaml      # Staged pan-organ CT training spec
├── docs/
│   ├── EXPERIMENTS.md              # Experiment logs with full results
│   ├── roadmap.md                  # Phase 1–6 execution plan
│   ├── hardware_setup.md           # ROCm/platform bootstrap
│   └── data_preprocessing.md       # DICOM → PNG pipeline details
├── zoo/                            # Model zoo package
│   ├── arch.py                     # PatchViT + ScaleEmbedding architecture
│   ├── hub.py                      # Model loading (local / HuggingFace Hub)
│   ├── encode.py                   # Zero-preprocessing inference API
│   ├── peft.py                     # LoRA adapter inject / save / load
│   ├── card.py                     # HuggingFace model card generator
│   ├── publish.py                  # HuggingFace Hub publishing pipeline
│   ├── data.py                     # Unified CT data loader (Manifest → DataLoader)
│   ├── models.py                   # Pydantic models (DatasetEntry, SliceMetadata, etc.)
│   ├── registry.py                 # YAML dataset catalog
│   ├── manifest.py                 # Parquet per-slice metadata
│   ├── lineage.py                  # Training provenance tracking
│   ├── merge.py                    # Multi-dataset weighted sampling
│   └── datasets/                   # Dataset YAML entries
│       ├── lidc_idri.yaml
│       └── pancreas_ct.yaml
├── scripts/
│   ├── phase5_big_run.py           # Main training script (DINO + Gram + Scale)
│   ├── evaluate_panorgan.py        # 6-metric evaluation suite
│   ├── finetune_lora.py            # LoRA fine-tuning training script
│   ├── phase2_preprocess_lidc_idri.py # DICOM → PNG preprocessor
│   ├── phase2_tcia_download.py     # TCIA dataset downloader
│   ├── mvp_combine_indices.py      # Multi-dataset index combiner
│   ├── extract_dicom_spacing.py    # DICOM spacing metadata extractor
│   ├── fetch_hf_data.sh            # Download processed data from HuggingFace
│   └── prep_remote_data.sh         # Cloud data prep pipeline (TCIA → HF)
├── tests/                          # 171 tests (zoo, card, publish, data, finetune)
├── runs/                           # Experiment artifacts (results, configs)
├── openspec/                       # Specifications and change proposals
├── requirements.in                 # Human-maintained dependencies
├── requirements.txt                # Pinned snapshot
└── README.md
```

## Hardware

DINO-X is designed to train on consumer hardware:

| Role | Hardware | Memory | Use |
|------|----------|--------|-----|
| Primary trainer | AMD Strix Halo | 128GB unified (~96GB VRAM) | Large model training |
| Cloud training | RTX 3090 / 4090 / 5070 | 12–24GB | MVP experiments |
| Preprocessing | Any GPU | — | DICOM conversion |

The AMD Strix Halo's 128GB unified memory breaks the "Memory Wall" that restricts
billion-parameter training to enterprise data centers.

## Criteria for Success

| Benchmark | Target | Status |
|-----------|--------|--------|
| View retrieval (label-free) | Beat random baseline | ✅ 14× on LIDC, 5× on Pancreas |
| Scale embedding ablation | Loss improvement | ✅ 67× lower loss |
| Dataset discrimination | AUC ≥ 0.95 | ✅ AUC = 1.000 |
| Spacing R² | ≥ 0.80 | ✅ R² = 0.876 |
| No feature collapse | Embedding σ > 0 | ✅ Verified |
| Linear probe AUC | > 0.90 on malignancy | Planned (Stage C) |
| Attention maps | Segment nodules unsupervised | Planned |

## License & Citation

- **License:** GPLv3 (see `LICENSE`)
- **Base Architecture:** Meta Research / DINOv3 with Gram Anchoring
- **Data:** LIDC-IDRI and Pancreas-CT from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)

> **Status:** Active development. Phases 1–5 infrastructure complete (ScaleEmbedding,
> data registry, training pipeline, model cards, HF Hub publishing, LoRA fine-tuning).
> MVP proven with 67× loss improvement on two-organ ablation. Pan-organ scaling
> (Stage A/B/C) and cross-modality expansion (MRI, X-ray) are next.

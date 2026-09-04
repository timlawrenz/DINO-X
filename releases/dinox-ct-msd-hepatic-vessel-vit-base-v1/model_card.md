---
license: apache-2.0
library_name: dinox-zoo
tags:
- medical-imaging
- ct
- self-supervised
- scale-aware
- dino-x
- hepatic-vessel
pipeline_tag: image-classification
language:
- en
---

# dinox-ct-msd-hepatic-vessel-vit-base-v1

Scale-aware ViT-Base (86M) organ-specialist for **hepatic / portal vessel tissue
detection in abdominal CT**, self-supervised on MSD Hepatic-Vessel, LoRA fine-tuned
for slice-level vessel-present classification.

**Research use only. Not a clinical diagnostic device.**

## Model card (summary)

| | |
|---|---|
| Architecture | Scale-aware ViT-Base (dim 768, depth 12, patch 14) |
| Pretraining | DINO self-supervised, single-organ MSD Hepatic-Vessel, 50K steps |
| Fine-tuning | LoRA (rank 8, alpha 16) on vessel-present binary labels |
| Internal AUROC | 0.9456 *(leakage-caveated, not release-supporting)* |
| **External AUROC (TCIA CRLM)** | **0.9413** slice-level / **0.739** patient-level, 197 patients |
| Input | `hu16_png` 16-bit HU slices, windowing level −30 / width 120, 224px |
| License | Apache-2.0 |

## Intended use
Slice-level detection of hepatic/portal vessel tissue in contrast-enhanced abdominal
CT, for research (backbone feature extraction, downstream fine-tuning). **Not** for
diagnosis or treatment planning.

## How to use the model

The release ships as **weights-only** — the inference harness is the open-source
[`zoo`](https://github.com/timlawrenz/DINO-X) package (this repo's codebase). This
card's numbers were produced by exactly this path (`scripts/eval_external.py`).

### Requirements

```bash
# Clone the codebase that defines the zoo package (PatchViT, loader, LoRA adapter)
git clone git@github.com:timlawrenz/DINO-X.git
cd DINO-X
# ship comes with a Python 3.14 venv; or create yours:
#   python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch, torchvision, peft, safetensors, pillow, nibabel
```

### 1. Load backbone + LoRA adapter + task head

```python
from pathlib import Path
import torch, torch.nn as nn
import numpy as np

# Point at the local clone of THIS HuggingFace repo (downloaded weights live here)
repo = Path("path/to/dinox-ct-msd-hepatic-vessel-vit-base-v1")

from zoo.hub import load_model, load_from_hub_dir
from zoo.peft import load_adapter

# (a) frozen scale-aware ViT-Base backbone from the repo's config.json + backbone.safetensors
backbone = load_from_hub_dir(repo, device="cuda")          # PatchViT, eval mode

# (b) LoRA adapter (rank 8, alpha 16) — PeftModel injected onto the backbone
backbone = load_adapter(backbone, repo)                    # reads adapter_config.json
backbone = backbone.to("cuda")

# (c) task head (nn.Linear 768 -> 2) + its trained weights
import json
ft = json.load(open(repo / "finetune_config.json"))
head = nn.Linear(backbone.dim, ft["num_classes"]).to("cuda")
head.load_state_dict(torch.load(repo / "head.pth", map_location="cuda"))

backbone.eval(); head.eval()
```

> **Note on `load_adapter`:** it calls `PeftModel.from_pretrained(backbone, adapter_dir)`
> and reads `rank`/`alpha`/`target_modules` straight from `adapter_config.json`. Do **not**
> also call a separate `apply_lora()` or the backbone gets double-wrapped and the adapter
> weights never attach — you'd silently score with a random LoRA.

### 2. Preprocess a slice (must match training exactly)

The model was trained on `hu16_png` 16-bit slices windowed to the liver HU range,
then resized/cropped to 224px with ImageNet normalization:

```python
from PIL import Image
from torchvision import transforms

WINDOW = {"level": -30.0, "width": 120.0}   # MUST match training: level -30, width 120
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def preprocess(png_path: str, spacing_mm: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    # 1) decode 16-bit PNG and un-encode to Hounsfield Units
    arr = np.array(Image.open(png_path), dtype=np.float32)
    if arr.ndim == 3: arr = arr[:, :, 0]
    hu = (arr - 32768.0) * 0.1                      # our hu16 encoding: HU = (u16 - 32768) * 0.1
    # 2) liver windowing to [0, 1]
    lo = WINDOW["level"] - WINDOW["width"] / 2.0
    x = np.clip((hu - lo) / WINDOW["width"], 0.0, 1.0)
    # 3) single slice -> 3 channels, then train-equivalent transform
    x = np.stack([x, x, x], axis=0)
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    x = tf(torch.from_numpy(x).contiguous())
    # 4) spacing must accompany the slice (the backbone is scale-aware)
    spacing = torch.tensor(list(spacing_mm), dtype=torch.float32)
    return x.unsqueeze(0).to("cuda"), spacing.to("cuda")   # (1,3,224,224), (3,)
```

### 3. Run inference

```python
@torch.no_grad()
def predict(png_path, spacing):
    x, sp = preprocess(png_path, spacing)
    logits = head(backbone(x, spacing=sp.unsqueeze(0))[:, 0])   # CLS token
    return torch.softmax(logits, dim=-1)[0, 1].item()           # P(vessel-present)

# Example from the repo's external-eval label format:
#   {'png_path': '...', 'spacing_x': 0.77, 'spacing_y': 0.77, 'spacing_z': 1.5}
score = predict("path/to/slice.png", (0.77, 0.77, 1.5))
print(f"P(vessel present) = {score:.3f}")
```

Output is `P(vessel present)` in `[0, 1]`. Decisions at a threshold belong to you —
this model ships **without calibration or uncertainty** (see Limitations).

### 4. Validate the load (recommended)

Re-run the external evaluation on TCIA CRLM with the exact reproduction harness to
confirm you can reproduce the headline number before trusting your own run:

```bash
python scripts/eval_external.py \
  --adapter path/to/dinox-ct-msd-hepatic-vessel-vit-base-v1 \
  --label-csv data/crlm/vessel_labels.csv \
  --window-level -30 --window-width 120 --device cuda
```

See `reproduce/README.md` for the full reproduction recipe and `reproduce/repro_metrics.py`.

## Construction (trust model)
This model ships with a full **evidence envelope** — every number is traceable to an
experiment, every failure is published, and the headline number reproduces
mechanically:

- **Why it exists & failures that shaped it:** `the_science/01_decision_trail.md`
- **Exact training recipe:** `the_science/02_training_recipe.md`
- **How we know the numbers are real (adversarial pass + external validation):** `the_science/03_validation.md`
- **Negative results:** `the_science/04_negative_results.md`
- **Known limitations:** `the_science/05_known_limits.md`
- **Reproduce it yourself:** `reproduce/README.md` + `reproduce/repro_metrics.py`

> We do not ask you to trust our AUROC. We give you the pan-organ failure, the record
> of us catching our own leakage, and a script to recompute the external number.

## Limitations
- Not a clinical device; no calibration / uncertainty.
- Single modality (CT), single task (per-slice vessel present).
- Internal AUROC has a pretraining-leakage caveat — cite the external number.

## Citation
*(to add)*
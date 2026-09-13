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
card's numbers were produced by exactly this path.

### Quick start (the one-liner)

Get `P(vessel-present)` for a CT slice in one call. Preprocessing (the trained
−30/120 HU window, resize, ImageNet norm, spacing) is baked in and cannot
diverge from the validated pipeline:

```python
from zoo.predict import load_classifier, predict_proba

clf = load_classifier("timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1")
p = predict_proba(clf, hu_slice, pixel_spacing=(0.77, 0.77), slice_thickness=1.5)
print(p)   # P(vessel present) in [0, 1]
```

`hu_slice` is a `(H, W)` numpy array in Hounsfield Units (e.g. a DICOM after
rescale slope/intercept). Output is a raw softmax probability — this model ships
**without calibration or uncertainty** (see Limitations).

### Requirements

```bash
git clone https://github.com/timlawrenz/DINO-X.git
cd DINO-X
python -m venv .venv && source .venv/bin/activate

# 1) torch for your hardware (CPU/CUDA default; see pytorch.org for CUDA/ROCm)
pip install torch torchvision
# 2) inference deps + the zoo package
pip install -r requirements-inference.txt
pip install .            # makes `from zoo.predict import ...` importable
```

> The model is device-agnostic and runs on CPU, CUDA, or ROCm.
> `requirements-inference.txt` has no machine-specific pins; the training repo's
> `requirements.txt` is only needed for the training hosts.

### Under the hood (what `load_classifier` does for you)

You don't need this to use the model — it's here so the load is transparent.
`load_classifier` assembles three artifacts from this repo in the correct order:

```python
# (a) frozen scale-aware ViT-Base backbone from config.json + backbone.safetensors
# (b) LoRA adapter (rank 8, alpha 16) injected via PeftModel.from_pretrained
# (c) task head (nn.Linear 768 -> 2) loaded from head.pth
```

It reads the trained HU window and `num_classes` from `finetune_config.json` so
preprocessing always matches training. **Do not** call a separate `apply_lora()`
on top — that double-wraps the backbone and silently scores with a random LoRA.

### Batch inference & the external-eval label format

For a list of slices, call `predict_proba` per slice (or `zoo.encode.encode_batch`
for backbone features). The external-eval label rows look like:

```
{'png_path': '...', 'spacing_x': 0.77, 'spacing_y': 0.77, 'spacing_z': 1.5, 'label': 1}
```

For 16-bit PNG inputs (the training `input_format`), pass
`input_format="hu16_png"` to `predict_proba`; it applies the same
`HU = (uint16 - 32768) * 0.1` decode then the trained window. The `hu_float`
and `hu16_png` paths agree to within float rounding.

### Validate the load (recommended)

Re-run the external evaluation on TCIA CRLM with the exact reproduction harness to
confirm you can reproduce the headline number before trusting your own run:

```bash
python scripts/eval_external.py \
  --adapter path/to/dinox-ct-msd-hepatic-vessel-vit-base-v1 \
  --label-csv data/crlm/vessel_labels.csv \
  --window-level -30 --window-width 120 --device cuda
```

> **Data access:** the CRLM evaluation set is derived from
> [TCIA ColoRectal Liver Metastases](https://www.cancerimagingarchive.net/) and is
> **not bundled** with this release in v1 (licensing review in progress). The
> reproduction commands above require obtaining CRLM from TCIA. A small bundled
> demo sample for verifying the inference mechanics ships in v1 — see
> `reproduce/README.md`.

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

If you use this model, please cite the model, the training dataset (MSD Hepatic
Vessel), and the external validation dataset (TCIA CRLM):

**This model**
```bibtex
@misc{lawrenz2026dinox_hepatic_vessel_v1,
  title   = {dinox-ct-msd-hepatic-vessel-vit-base-v1: Scale-aware ViT-Base specialist
             for hepatic vessel detection in abdominal CT (DINO-X model zoo)},
  author  = {Lawrenz, Tim},
  year    = {2026},
  howpublished = {\url{https://huggingface.co/timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1}},
  note    = {Release with evidence envelope: external AUROC 0.9413 (TCIA CRLM, 197 patients)}
}
```

**Training data — Medical Segmentation Decathlon (Task 08: Hepatic Vessels)**
```bibtex
@article{antonelli2022msd,
  title   = {The Medical Segmentation Decathlon},
  author  = {Antonelli, Michela and Reinke, Annika and Bakas, Spyridon and Farahani, Keyvan
             and Kopp-Schneider, Annette and Landman, Bennett A. and Litjens, Geert and
             Menze, Bjoern and Ronneberger, Olaf and Summers, Ronald M. and others},
  journal = {Nature Communications},
  year    = {2022},
  volume  = {13},
  pages   = {4128},
  doi     = {10.1038/s41467-022-30695-9},
  note    = {arXiv:2106.05735}
}
```

**External validation data — TCIA Colorectal-Liver-Metastases**
```bibtex
@misc{simpson2023crlm,
  title   = {Preoperative CT and Survival Data for Patients Undergoing Resection of
             Colorectal Liver Metastases (Colorectal-Liver-Metastases) (Version 2) [Data set]},
  author  = {Simpson, Amber L. and Peoples, Jacob and Creasy, John M. and Fichtinger, Gabor
             and Gangai, Natalie and Lasso, Andras and Keshava Murthy, Keshava N. and
             Shia, Jinru and D'Angelica, Michael I. and Do, Richard K. G.},
  year    = {2023},
  publisher = {The Cancer Imaging Archive},
  doi     = {10.7937/QXK2-QG03}
}
```
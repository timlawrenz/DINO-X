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
| Input | `hu16_png` 16-bit HU slices, lung window (level -30, width 120), 224px |
| License | Apache-2.0 |

## Intended use
Slice-level detection of hepatic/portal vessel tissue in contrast-enhanced abdominal
CT, for research (backbone feature extraction, downstream fine-tuning). **Not** for
diagnosis or treatment planning.

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
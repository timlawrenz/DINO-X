# Data Preprocessing Strategy: LIDC-IDRI vs CIFAR-10

This document details the exact preprocessing steps used in the DINO-X project, contrasting the custom pipeline developed for LIDC-IDRI Lung CTs with the standard pipeline used for the CIFAR-10 baseline.

## 1. LIDC-IDRI Pipeline (Medical CT)

Our goal is to preserve the full dynamic range of the CT scan (Hounsfield Units) while making the data consumable by standard computer vision models (which expect 0-1 or normalized floats).

### Step 1: Ingestion (DICOM to Raw PNG)
**Script:** `scripts/preprocessing/phase2_preprocess_lidc_idri.py`

We convert 12-bit DICOM files into 16-bit PNGs to create a "Raw" dataset that is fast to load but semantically lossless.

1.  **Read DICOM:** Load pixel array using `pydicom`.
2.  **Apply Rescale Slope/Intercept:** Convert raw integer values to **Hounsfield Units (HU)**.
    *   Air $\approx$ -1000 HU
    *   Water = 0 HU
    *   Bone $\approx$ +400 - +1000 HU
3.  **Clip:** Range is clipped to `[-1000, 4000]`.
    *   Reason: Values below -1000 are sensor artifacts/noise. Values above 4000 are metal artifacts or dense bone, irrelevant for soft tissue analysis.
4.  **Offset & Cast:** Shift values to be positive for `uint16` storage.
    *   $PV_{png} = \text{round}(HU + 32768)$
    *   Stored as single-channel 16-bit PNG.

### Step 2: Dynamic Loading (PNG to Tensor)
**Script:** `scripts/phase5_big_run.py` (`PngDataset`)

During training, we load these 16-bit PNGs and apply **Random Windowing** to simulate the way radiologists view scans.

1.  **Load:** Read 16-bit PNG, convert to float.
2.  **Restore HU:** $HU = \text{pixel} - 32768$
3.  **Random Windowing (Intensity Augmentation):**
    *   Select a random **Level** ($L$) from `[-400, 400]` (Soft tissue to Lung focus).
    *   Select a random **Width** ($W$) from `[800, 2000]` (Wide dynamic range).
    *   Apply Windowing: $x = \frac{HU - (L - W/2)}{W}$
    *   Clip to `[0, 1]`.
4.  **Spatial Context (2.5D Stacking):**
    *   Load slice $z$, but also $z-1$ and $z+1$.
    *   Stack into an RGB-like image: `(3, H, W)`.
    *   Channel 0 = Upper slice, Channel 1 = Center slice, Channel 2 = Lower slice.
5.  **Spatial Augmentation (The "Fix"):**
    *   **RandomResizedCrop:** Crop a random portion (scale 0.5-1.0) and resize to $224 \times 224$.
    *   **RandomHorizontalFlip:** Left/Right symmetry augmentation.
6.  **Normalization:**
    *   Apply ImageNet statistics (Mean: 0.485..., Std: 0.229...).
    *   This shifts the `0..1` input to approximately `[-2, 2]`.

## 2. CIFAR-10 Baseline Pipeline (Natural Images)

**Script:** `scripts/baseline_cifar10_pretrain.py`

This follows the standard self-supervised learning recipe (SimCLR/DINO).

1.  **Load:** Standard RGB images ($32 \times 32$).
2.  **Augmentations:**
    *   **RandomResizedCrop:** Scale 0.5-1.0.
    *   **HorizontalFlip.**
    *   **ColorJitter:** Brightness, contrast, saturation, hue (0.4/0.4/0.2/0.1).
    *   **RandomGrayscale:** Drop color channels (p=0.2).
    *   **Solarization** (sometimes used, not in our minimal baseline).
3.  **Normalization:**
    *   Mean: `(0.4914, 0.4822, 0.4465)`
    *   Std: `(0.2470, 0.2435, 0.2616)`

## 3. Comparison Summary

| Feature | CIFAR-10 Baseline | LIDC-IDRI (DINO-X) | Reason for Difference |
| :--- | :--- | :--- | :--- |
| **Source** | 8-bit RGB (Natural) | 12-bit DICOM (Medical) | Hardware differences (Camera vs CT Detector). |
| **Input Shape** | $32 \times 32 \times 3$ | $224 \times 224 \times 3$ | CT resolution is much higher; ViT needs patches. |
| **Channels** | R, G, B (Color) | $z-1, z, z+1$ (Volumetric) | CT is grayscale; we use channels for 3D context. |
| **Intensity Aug** | Color Jitter (Hue/Sat) | Random Windowing (Level/Width) | "Color" in CT is physical density. Windowing is the radiological equivalent of lighting changes. |
| **Spatial Aug** | Crop + Flip | Crop + Flip | **Crucial:** Both domains need spatial invariance to learn semantic features. |
| **Normalization** | Dataset-specific (Static) | ImageNet (Static) | We use ImageNet norms to keep gradients stable, even though data is not natural images. |

## 4. Why "Random Windowing" matters
In natural images, a red car is always red. Changing the hue makes it a "different" car.
In CT, a tumor has a fixed density (HU). However, depending on whether the doctor is looking for "bone detail" or "soft tissue detail", they map those HU values to brightness differently.
By randomly varying the window, we force the model to learn that **density X** is the feature, regardless of whether it appears white, gray, or black in the current view.

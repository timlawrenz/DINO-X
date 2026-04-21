"""Zero-preprocessing encode() API for DINO-X models.

Clinical researchers pass raw data + spacing, get features back.
No complex dependencies, no resampling, no manual windowing.

Example::

    from zoo.hub import load_model
    from zoo.encode import encode

    model = load_model("timlawrenz/dinox-ct-vit-small-v1")
    features = encode(
        model,
        image=raw_dicom_array,          # HxW uint16 or float32
        pixel_spacing=(0.5, 0.5),       # mm, from DICOM header
        slice_thickness=1.0,            # mm, from DICOM header
    )
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from zoo.arch import PatchViT

# ImageNet normalization (matches training pipeline)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _hu_window(
    arr: np.ndarray,
    level: float = 40.0,
    width: float = 400.0,
) -> np.ndarray:
    """Apply HU windowing to a float array, returning [0, 1] range."""
    lower = level - width / 2
    upper = level + width / 2
    arr = np.clip(arr, lower, upper)
    return (arr - lower) / (upper - lower)


def _to_hu(
    arr: np.ndarray,
    input_format: str,
) -> np.ndarray:
    """Convert input array to Hounsfield Units (float32)."""
    if input_format == "hu_float":
        return arr.astype(np.float32)
    elif input_format == "hu16_png":
        # Our 16-bit PNG encoding: HU = (uint16 - 32768) * 0.1
        return (arr.astype(np.float32) - 32768.0) * 0.1
    elif input_format == "windowed_float":
        # Already windowed to [0, 1] — skip HU conversion
        return arr.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown input_format: '{input_format}'. "
            "Supported: 'hu_float', 'hu16_png', 'windowed_float'"
        )


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize 2D array to (size, size) using PIL for quality."""
    from PIL import Image

    img = Image.fromarray(arr)
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def _normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a (C, H, W) tensor."""
    mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def encode(
    model: PatchViT,
    image: np.ndarray,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
    *,
    input_format: Literal["hu_float", "hu16_png", "windowed_float"] = "hu_float",
    hu_level: float = 40.0,
    hu_width: float = 400.0,
    return_all_tokens: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Encode a medical image into features using a DINO-X backbone.

    Handles all preprocessing internally: format conversion, HU windowing,
    resizing, normalization, and spacing injection.

    Args:
        model: A PatchViT backbone (from ``load_model()``).
        image: Input image as numpy array.
            - Shape ``(H, W)`` for a single slice (replicated to 3 channels).
            - Shape ``(H, W, 3)`` or ``(3, H, W)`` for 3-slice context (z-1, z, z+1).
        pixel_spacing: ``(spacing_x, spacing_y)`` in mm from DICOM header.
        slice_thickness: Slice thickness in mm from DICOM header.
        input_format: How to interpret the pixel values:
            - ``"hu_float"``: Already in Hounsfield Units (float). Default.
            - ``"hu16_png"``: 16-bit PNG encoding ``HU = (uint16 - 32768) * 0.1``.
            - ``"windowed_float"``: Already windowed to [0, 1] range.
        hu_level: HU window center (ignored if ``input_format="windowed_float"``).
        hu_width: HU window width (ignored if ``input_format="windowed_float"``).
        return_all_tokens: If True, return all tokens ``(1, N+1, dim)``.
            If False (default), return only CLS token ``(1, dim)``.
        device: Device for inference. If None, uses model's device.

    Returns:
        Feature tensor: ``(1, dim)`` CLS features, or ``(1, N+1, dim)`` all tokens.

    Example::

        features = encode(model, raw_array, pixel_spacing=(0.7, 0.7), slice_thickness=1.5)
    """
    if device is None:
        device = next(model.parameters()).device

    img_size = model.img_size

    # --- Step 1: Convert to HU or windowed float ---
    if input_format == "windowed_float":
        # Already [0, 1]; shape (H, W) or (H, W, 3) or (3, H, W)
        arr = image.astype(np.float32)
    else:
        arr = _to_hu(image, input_format)
        arr = _hu_window(arr, level=hu_level, width=hu_width)

    # --- Step 2: Handle channel dimension ---
    if arr.ndim == 2:
        # Single slice → replicate to 3 channels
        channels = [arr, arr, arr]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # (H, W, 3) → split channels
        channels = [arr[:, :, i] for i in range(3)]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        # (3, H, W) → split channels
        channels = [arr[i] for i in range(3)]
    else:
        raise ValueError(
            f"Unsupported image shape: {arr.shape}. "
            "Expected (H, W), (H, W, 3), or (3, H, W)."
        )

    # --- Step 3: Resize each channel ---
    resized = [_resize(ch, img_size) for ch in channels]
    tensor = torch.tensor(np.stack(resized, axis=0), dtype=torch.float32)  # (3, H, W)

    # --- Step 4: Normalize ---
    tensor = _normalize_imagenet(tensor)

    # --- Step 5: Batch dimension ---
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    # --- Step 6: Spacing tensor ---
    spacing: torch.Tensor | None = None
    if model.scale_aware:
        spacing = torch.tensor(
            [[pixel_spacing[0], pixel_spacing[1], slice_thickness]],
            dtype=torch.float32,
            device=device,
        )

    # --- Step 7: Forward pass ---
    with torch.no_grad():
        features = model(tensor, spacing=spacing)

    if return_all_tokens:
        return features  # (1, N+1+registers, dim)

    # Return CLS token only
    return features[:, 0:1, :]  # (1, 1, dim) → squeeze to (1, dim)


def encode_batch(
    model: PatchViT,
    images: list[np.ndarray],
    spacings: list[tuple[float, float, float]],
    *,
    input_format: Literal["hu_float", "hu16_png", "windowed_float"] = "hu_float",
    hu_level: float = 40.0,
    hu_width: float = 400.0,
    return_all_tokens: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Encode a batch of medical images.

    Args:
        model: A PatchViT backbone.
        images: List of numpy arrays, each (H, W) or (H, W, 3).
        spacings: List of (spacing_x, spacing_y, slice_thickness) tuples.
        input_format: See ``encode()``.
        hu_level: HU window center.
        hu_width: HU window width.
        return_all_tokens: If True, return all tokens per image.
        device: Device for inference.

    Returns:
        Feature tensor: ``(B, dim)`` CLS features, or ``(B, N+1, dim)`` all tokens.
    """
    if len(images) != len(spacings):
        raise ValueError(
            f"images ({len(images)}) and spacings ({len(spacings)}) must have same length"
        )

    results = []
    for img, (sx, sy, st) in zip(images, spacings):
        feat = encode(
            model, img,
            pixel_spacing=(sx, sy),
            slice_thickness=st,
            input_format=input_format,
            hu_level=hu_level,
            hu_width=hu_width,
            return_all_tokens=return_all_tokens,
            device=device,
        )
        results.append(feat)

    return torch.cat(results, dim=0)

"""Stupid-simple, one-line classification for released DINO-X specialist models.

This module exists so a researcher who finds a DINO-X model on HuggingFace can
go from "model ID" to "P(task-positive)" in a single call, WITHOUT knowing
about LoRA adapters, task heads, or HU windowing. It bakes in the exact
preprocessing the model was trained and validated with, so there is ONE
preprocessing path and it is the correct one.

Design rules (learned the hard way — see the model card's load_adapter warning):
  * Load backbone + LoRA adapter + task head HERE, once, in the right order.
    Do not call apply_lora() separately (that double-wraps and silently scores
    with a random LoRA).
  * The HU window is pinned to the model's training window, NOT the generic
    encode() default. For dinox-ct-msd-hepatic-vessel-vit-base-v1 that is
    level=-30, width=120 (liver window). Using any other window changes the
    number.

Example — one line to a probability::

    from zoo.predict import load_classifier, predict_proba

    clf = load_classifier("timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1")
    p = predict_proba(clf, hu_slice, pixel_spacing=(0.77, 0.77), slice_thickness=1.5)
    print(p)   # P(vessel-present) in [0, 1]

``hu_slice`` is a (H, W) numpy array of Hounsfield Units (float), e.g. straight
from a DICOM after applying rescale slope/intercept.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from zoo.hub import load_model

logger = logging.getLogger(__name__)

# ImageNet normalization (matches the training pipeline exactly).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Fallback trained window for this release. The authoritative value SHOULD live
# in the model's own metadata (see ``window`` in ``load_classifier``); this is
# the documented training window for the hepatic-vessel v1 release and is used
# only if the model repo does not declare one.
_DEFAULT_WINDOW = {"level": -30.0, "width": 120.0}


@dataclass
class DinoXClassifier:
    """A fully-assembled, frozen, eval-mode classifier ready for inference.

    Holds the LoRA-adapted backbone and the task head plus the exact
    preprocessing parameters the model was trained with. Produced by
    :func:`load_classifier`; do not assemble by hand.
    """

    backbone: nn.Module          # PeftModel-wrapped PatchViT (LoRA injected)
    head: nn.Linear              # task head (dim -> num_classes)
    window_level: float
    window_width: float
    num_classes: int
    img_size: int
    scale_aware: bool
    device: torch.device
    model_id: str = ""

    def predict_tensor(self, image_3chw: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        """Forward a preprocessed (1,3,H,W) tensor + spacing (1,3) -> logits."""
        image_3chw = image_3chw.to(self.device)
        spacing = spacing.to(self.device) if self.scale_aware else None
        with torch.no_grad():
            feats = self.backbone(image_3chw, spacing=spacing)
            cls = feats[:, 0] if feats.ndim == 3 else feats
            logits = self.head(cls)
        return logits


def _unwrap_base(model: nn.Module) -> nn.Module:
    """Return the underlying PatchViT from a possibly-PEFT-wrapped model."""
    # PeftModel exposes .base_model (an LoraModel) -> .model (the PatchViT).
    m = model
    for attr in ("base_model", "model", "module"):
        inner = getattr(m, attr, None)
        if inner is not None and inner is not m:
            # descend while wrappers exist
            candidate = inner
            # Peel common wrapper chain
            seen = set()
            while id(candidate) not in seen:
                seen.add(id(candidate))
                nxt = None
                for a in ("model", "module"):
                    if hasattr(candidate, a) and getattr(candidate, a) is not candidate:
                        nxt = getattr(candidate, a)
                        break
                if nxt is None:
                    break
                candidate = nxt
            return candidate
    return m


def _read_window(model_dir: Path | None) -> tuple[float, float]:
    """Read the trained HU window from the model repo if declared, else default.

    Looks for ``window_level``/``window_width`` (or a nested ``window`` dict) in
    ``finetune_config.json``. Falls back to the documented v1 training window.
    """
    if model_dir is not None:
        ft_path = model_dir / "finetune_config.json"
        if ft_path.exists():
            try:
                ft = json.loads(ft_path.read_text())
                win = ft.get("window")
                if isinstance(win, dict) and "level" in win and "width" in win:
                    return float(win["level"]), float(win["width"])
                if "window_level" in ft and "window_width" in ft:
                    return float(ft["window_level"]), float(ft["window_width"])
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Could not parse window from %s: %s", ft_path, e)
    return _DEFAULT_WINDOW["level"], _DEFAULT_WINDOW["width"]


def load_classifier(
    model_id_or_path: str,
    *,
    device: str | torch.device | None = None,
) -> DinoXClassifier:
    """Load a released DINO-X specialist as a ready-to-use classifier.

    Assembles backbone + LoRA adapter + task head from a HuggingFace Hub model
    ID or a local hub-format directory, freezes everything, sets eval mode, and
    reads the exact preprocessing the model was trained with.

    Args:
        model_id_or_path: HF Hub model ID (``timlawrenz/dinox-...``) or a local
            directory containing ``config.json``, ``backbone.safetensors``,
            ``adapter_config.json``, ``adapter_model.safetensors``, ``head.pth``,
            and ``finetune_config.json``.
        device: Target device. Defaults to CUDA if available else CPU.

    Returns:
        A :class:`DinoXClassifier` ready for :func:`predict_proba`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Resolve a local dir so we can read the sibling files (head, adapter, ft cfg).
    p = Path(model_id_or_path)
    if p.is_dir():
        local_dir = p
    else:
        from huggingface_hub import snapshot_download

        local_dir = Path(snapshot_download(model_id_or_path))

    # (1) backbone
    backbone = load_model(str(local_dir), device=device)

    # (2) LoRA adapter — the ONLY wrap. Do not apply_lora() anywhere else.
    from zoo.peft import load_adapter

    backbone = load_adapter(backbone, local_dir)
    backbone = backbone.to(device).eval()

    # (3) task head, reading dims defensively from the unwrapped base
    ft = json.loads((local_dir / "finetune_config.json").read_text())
    num_classes = int(ft.get("num_classes", 2))
    base = _unwrap_base(backbone)
    dim = getattr(base, "dim", None) or getattr(backbone, "dim", None)
    if dim is None:
        # last resort: infer from config.json
        cfg = json.loads((local_dir / "config.json").read_text())
        dim = int(cfg.get("dim", 768))
    head = nn.Linear(int(dim), num_classes).to(device)
    head.load_state_dict(torch.load(local_dir / "head.pth", map_location=device))
    head.eval()

    level, width = _read_window(local_dir)
    img_size = int(getattr(base, "img_size", 224))
    scale_aware = bool(getattr(base, "scale_aware", True))

    logger.info(
        "Loaded classifier %s: dim=%d classes=%d window=(%.0f,%.0f) device=%s",
        model_id_or_path, dim, num_classes, level, width, device,
    )
    return DinoXClassifier(
        backbone=backbone,
        head=head,
        window_level=level,
        window_width=width,
        num_classes=num_classes,
        img_size=img_size,
        scale_aware=scale_aware,
        device=device,
        model_id=str(model_id_or_path),
    )


def _to_hu(arr: np.ndarray, input_format: str) -> np.ndarray:
    """Convert an input array to Hounsfield Units (float32)."""
    if input_format == "hu_float":
        return arr.astype(np.float32)
    if input_format == "hu16_png":
        # 16-bit PNG encoding: HU = (uint16 - 32768) * 0.1
        return (arr.astype(np.float32) - 32768.0) * 0.1
    if input_format == "windowed_float":
        return arr.astype(np.float32)
    raise ValueError(
        f"Unknown input_format '{input_format}'. "
        "Supported: 'hu_float', 'hu16_png', 'windowed_float'."
    )


def _preprocess(
    clf: DinoXClassifier,
    image: np.ndarray,
    *,
    input_format: str,
) -> torch.Tensor:
    """HU -> trained window -> resize -> ImageNet norm -> (1,3,H,W) tensor.

    Uses the classifier's OWN trained window — never a caller-supplied one —
    so preprocessing cannot silently diverge from the validated pipeline.
    """
    from PIL import Image

    arr = _to_hu(image, input_format)
    if input_format != "windowed_float":
        lo = clf.window_level - clf.window_width / 2.0
        arr = np.clip((arr - lo) / clf.window_width, 0.0, 1.0)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a single 2D slice, got shape {arr.shape}")

    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    img = img.resize((clf.img_size, clf.img_size), resample)
    x = np.array(img, dtype=np.float32) / 255.0

    t = torch.from_numpy(np.stack([x, x, x], axis=0)).contiguous()  # (3,H,W)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)  # (1,3,H,W)


@torch.no_grad()
def predict_proba(
    clf: DinoXClassifier,
    image: np.ndarray,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
    *,
    input_format: Literal["hu_float", "hu16_png", "windowed_float"] = "hu_float",
    positive_class: int = 1,
) -> float:
    """Return P(task-positive) for a single CT slice, in [0, 1].

    This is the stupid-simple entry point. Preprocessing (HU windowing at the
    model's trained window, resize, ImageNet norm, spacing injection) is handled
    internally and cannot diverge from the validated pipeline.

    Args:
        clf: A :class:`DinoXClassifier` from :func:`load_classifier`.
        image: (H, W) numpy array. By default interpreted as Hounsfield Units
            (``input_format='hu_float'``), e.g. from a DICOM after rescale.
        pixel_spacing: ``(spacing_x, spacing_y)`` in mm from the DICOM header.
        slice_thickness: Slice thickness in mm from the DICOM header.
        input_format: How to interpret pixel values (see :func:`_to_hu`).
        positive_class: Index of the positive class (default 1).

    Returns:
        Probability of the positive class as a float in [0, 1].

    Note:
        This model ships WITHOUT calibration or uncertainty. The returned value
        is a raw softmax probability, not a calibrated confidence.
    """
    x = _preprocess(clf, image, input_format=input_format)
    spacing = torch.tensor(
        [[pixel_spacing[0], pixel_spacing[1], slice_thickness]], dtype=torch.float32
    )
    logits = clf.predict_tensor(x, spacing)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0, positive_class].item())

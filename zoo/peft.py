"""Parameter-Efficient Fine-Tuning (PEFT) support for DINO-X models.

Thin wrapper around HuggingFace ``peft`` library for injecting LoRA
adapters into PatchViT backbones. Automatically freezes physical
scale embeddings and base tokenization to prevent downstream users
from accidentally overwriting learned physics.

Example::

    from zoo.hub import load_model
    from zoo.peft import apply_lora, save_adapter, load_adapter

    model = load_model("timlawrenz/dinox-ct-vit-small-v1")
    model = apply_lora(model, rank=8)

    # ... train on local hospital data ...

    save_adapter(model, "my-pe-adapter/")

    # Later:
    model = load_model("timlawrenz/dinox-ct-vit-small-v1")
    model = load_adapter(model, "my-pe-adapter/")
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn

from zoo.arch import PatchViT

logger = logging.getLogger(__name__)

# Default modules to target for LoRA injection.
# These match the timm-style naming in zoo.arch:
#   Attention: qkv, proj
#   Mlp: fc1, fc2
DEFAULT_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]

# Modules that must ALWAYS be frozen during LoRA fine-tuning.
# The ScaleEmbedding encodes physical spacing — adapters should learn
# pathology features, not alternate physics.
_FROZEN_PREFIXES = (
    "scale_embed.",   # Physical spacing MLP — never retrain
    "patch_embed.",   # Base visual tokenization
)

_FROZEN_NAMES = (
    "cls_token",      # CLS token
    "pos_embed",      # Positional embeddings
    "registers",      # Register tokens
)


def apply_lora(
    model: PatchViT,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
    modules_to_save: list[str] | None = None,
) -> nn.Module:
    """Inject LoRA adapters into a PatchViT backbone.

    Freezes all base parameters, then unfreezes only the injected LoRA
    weights. Physical infrastructure (ScaleEmbedding, patch_embed, positional
    embeddings) is **always** frozen regardless of other settings.

    Args:
        model: A PatchViT backbone (from ``load_model()``).
        rank: LoRA rank (number of low-rank dimensions). Default 8.
        alpha: LoRA scaling factor. Default 16.
        target_modules: Which linear layers to wrap. Default:
            ``["qkv", "proj", "fc1", "fc2"]`` (all attention + MLP linears).
        dropout: LoRA dropout rate. Default 0.05.
        modules_to_save: Additional modules to keep trainable (e.g., a
            classification head). Default None.

    Returns:
        The model wrapped with LoRA adapters. Only adapter parameters
        have ``requires_grad=True``.

    Raises:
        ImportError: If ``peft`` is not installed.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )

    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=modules_to_save,
    )

    wrapped = get_peft_model(model, config)

    # Hard-freeze physical infrastructure — no exceptions
    _freeze_physical_layers(wrapped)

    total = sum(p.numel() for p in wrapped.parameters())
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

    logger.info(
        "LoRA applied: rank=%d, alpha=%.1f, targets=%s",
        rank, alpha, target_modules,
    )
    logger.info(
        "Parameters: %d total, %d trainable (%.1f%%)",
        total, trainable, 100.0 * trainable / total if total else 0,
    )

    return wrapped


def _freeze_physical_layers(model: nn.Module) -> None:
    """Freeze ScaleEmbedding, patch_embed, and positional parameters.

    This is the clinical safety guardrail: downstream LoRA adapters
    learn pathology features, not alternate physics.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        should_freeze = any(prefix in name for prefix in _FROZEN_PREFIXES)
        should_freeze = should_freeze or any(n in name for n in _FROZEN_NAMES)

        if should_freeze and param.requires_grad:
            param.requires_grad = False
            frozen_count += 1

    if frozen_count > 0:
        logger.info("Froze %d physical infrastructure parameters", frozen_count)


def save_adapter(
    model: nn.Module,
    output_dir: str | Path,
) -> Path:
    """Save only the LoRA adapter weights.

    Creates a directory with adapter config and weights that can be
    loaded later with ``load_adapter()``.

    Args:
        model: A peft-wrapped model (from ``apply_lora()``).
        output_dir: Directory to save the adapter.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    logger.info("Saved LoRA adapter to %s", out)
    return out


def load_adapter(
    model: PatchViT,
    adapter_path: str | Path,
) -> nn.Module:
    """Load a saved LoRA adapter onto a PatchViT backbone.

    Args:
        model: A base PatchViT backbone (from ``load_model()``).
        adapter_path: Path to directory saved by ``save_adapter()``.

    Returns:
        The model with LoRA adapter loaded and physical layers frozen.

    Raises:
        ImportError: If ``peft`` is not installed.
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        )

    adapter_path = Path(adapter_path)
    wrapped = PeftModel.from_pretrained(model, str(adapter_path))
    _freeze_physical_layers(wrapped)

    logger.info("Loaded LoRA adapter from %s", adapter_path)
    return wrapped


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters.

    Returns:
        Dict with ``total``, ``trainable``, and ``frozen`` counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }

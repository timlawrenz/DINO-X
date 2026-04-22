"""Model loading from local checkpoints or HuggingFace Hub.

Supports two checkpoint formats:

1. **Training checkpoint** (``.pth``): Contains ``student``/``teacher``
   state dicts, optimizer, config, etc. Produced by ``phase5_big_run.py``.
2. **Hub checkpoint**: Contains ``backbone`` state dict + ``config.json``.
   The minimal format for distribution via HuggingFace Hub.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from zoo.arch import DinoStudentTeacher, PatchViT, migrate_state_dict, needs_migration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Default ViT-Small config (matches training defaults)
DEFAULT_CONFIG: dict[str, Any] = {
    "img_size": 224,
    "patch": 16,
    "dim": 384,
    "depth": 6,
    "heads": 6,
    "mlp_ratio": 4.0,
    "num_registers": 4,
    "scale_aware": False,
    "out_dim": 8192,
}


def _build_backbone(config: dict[str, Any]) -> PatchViT:
    """Instantiate a PatchViT from a config dict."""
    return PatchViT(
        img_size=config.get("img_size", 224),
        patch=config.get("patch", 16),
        dim=config.get("dim", 384),
        depth=config.get("depth", 6),
        heads=config.get("heads", 6),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        num_registers=config.get("num_registers", 4),
        scale_aware=config.get("scale_aware", False),
    )


# ---------------------------------------------------------------------------
# Loading from training checkpoint (.pth)
# ---------------------------------------------------------------------------


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Remove a key prefix (e.g. 'backbone.') from all matching keys."""
    new_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_from_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    config_override: dict[str, Any] | None = None,
) -> PatchViT:
    """Load a PatchViT backbone from a training checkpoint.

    Extracts the student backbone weights, stripping the ``backbone.``
    prefix and ``head.*`` keys from the DinoStudentTeacher wrapper.
    Automatically migrates old-format state dict keys if needed.

    Args:
        path: Path to ``.pth`` checkpoint.
        device: Target device.
        config_override: Override config values (useful when checkpoint
            lacks config or you want to change settings).

    Returns:
        A PatchViT model loaded with the backbone weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = dict(DEFAULT_CONFIG)
    if "config" in payload:
        ckpt_config = payload["config"]
        if isinstance(ckpt_config, dict):
            # Training checkpoints nest model config under "model" key
            if "model" in ckpt_config and isinstance(ckpt_config["model"], dict):
                config.update(ckpt_config["model"])
            # Also pick up top-level config keys (img_size, scale_aware, etc.)
            for k in ("img_size", "scale_aware"):
                if k in ckpt_config:
                    config[k] = ckpt_config[k]
    if config_override:
        config.update(config_override)

    backbone = _build_backbone(config)

    # Get student state dict and extract backbone weights
    if "student" in payload:
        sd = payload["student"]
    elif "model" in payload:
        sd = payload["model"]
    else:
        sd = payload

    # Migrate old-format keys if needed
    if needs_migration(sd):
        logger.info("Migrating old-format state dict keys to timm-style")
        sd = migrate_state_dict(sd)

    # Strip wrapper prefixes
    if any(k.startswith("backbone.") for k in sd):
        sd = _strip_prefix(sd, "backbone.")

    # Remove projection head keys
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    # Handle scale_aware mismatch
    if not config.get("scale_aware", False):
        sd = {k: v for k, v in sd.items() if not k.startswith("scale_embed.")}

    backbone.load_state_dict(sd, strict=False)
    backbone.eval()

    logger.info(
        "Loaded backbone from training checkpoint: %s (dim=%d, depth=%d, scale_aware=%s)",
        path.name, config["dim"], config["depth"], config.get("scale_aware", False),
    )
    return backbone


# ---------------------------------------------------------------------------
# Loading from Hub format (backbone + config.json)
# ---------------------------------------------------------------------------


def load_from_hub_dir(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PatchViT:
    """Load a PatchViT from a hub-format directory.

    Expected structure::

        model_dir/
        ├── config.json
        └── backbone.pth  (or backbone.safetensors)

    Args:
        model_dir: Path to directory with config.json and weights.
        device: Target device.

    Returns:
        A PatchViT model in eval mode.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    config = json.loads(config_path.read_text())
    backbone = _build_backbone(config)

    # Try safetensors first, then .pth
    safetensors_path = model_dir / "backbone.safetensors"
    pth_path = model_dir / "backbone.pth"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
            sd = load_file(str(safetensors_path), device=str(device))
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            )
    elif pth_path.exists():
        sd = torch.load(pth_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No weights found in {model_dir}. "
            f"Expected backbone.safetensors or backbone.pth"
        )

    if needs_migration(sd):
        sd = migrate_state_dict(sd)

    backbone.load_state_dict(sd, strict=True)
    backbone.eval()

    logger.info("Loaded backbone from hub dir: %s", model_dir)
    return backbone


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


def load_model(
    model_id_or_path: str,
    *,
    device: str | torch.device = "cpu",
    config_override: dict[str, Any] | None = None,
) -> PatchViT:
    """Load a DINO-X backbone from a local path or HuggingFace Hub.

    Accepts:
    - A local ``.pth`` file (training checkpoint format).
    - A local directory with ``config.json`` + ``backbone.pth`` (hub format).
    - A HuggingFace Hub model ID (e.g., ``timlawrenz/dinox-ct-vit-small-v1``).

    Args:
        model_id_or_path: Local path or HuggingFace Hub model ID.
        device: Target device for the loaded model.
        config_override: Override config values.

    Returns:
        A PatchViT model in eval mode.

    Example::

        model = load_model("timlawrenz/dinox-ct-vit-small-v1")
        model = load_model("checkpoints/step_5000.pth", device="cuda")
    """
    p = Path(model_id_or_path)

    # Case 1: Local .pth file (training checkpoint)
    if p.is_file() and p.suffix == ".pth":
        return load_from_training_checkpoint(p, device=device, config_override=config_override)

    # Case 2: Local hub-format directory
    if p.is_dir() and (p / "config.json").exists():
        return load_from_hub_dir(p, device=device)

    # Case 3: HuggingFace Hub model ID
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            f"Cannot load '{model_id_or_path}': not a local file/directory, "
            "and huggingface_hub is not installed. "
            "Install with: pip install huggingface_hub"
        )

    logger.info("Downloading model from HuggingFace Hub: %s", model_id_or_path)
    local_dir = snapshot_download(model_id_or_path)
    return load_from_hub_dir(local_dir, device=device)


# ---------------------------------------------------------------------------
# Export to hub format
# ---------------------------------------------------------------------------


def export_hub_checkpoint(
    backbone: PatchViT,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    use_safetensors: bool = False,
) -> Path:
    """Export a PatchViT backbone to hub-distributable format.

    Creates ``config.json`` and ``backbone.pth`` (or ``.safetensors``)
    in the output directory.

    Args:
        backbone: The PatchViT model to export.
        output_dir: Directory to write files into.
        config: Model config dict. If None, inferred from model attributes.
        use_safetensors: If True, save as safetensors (requires safetensors package).

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build config from model attributes if not provided
    if config is None:
        config = {
            "img_size": backbone.img_size,
            "patch": backbone.patch,
            "dim": backbone.dim,
            "depth": len(backbone.blocks),
            "heads": backbone.blocks[0].attn.num_heads,
            "mlp_ratio": 4.0,
            "num_registers": backbone.num_registers,
            "scale_aware": backbone.scale_aware,
        }

    (out / "config.json").write_text(json.dumps(config, indent=2))

    sd = backbone.state_dict()
    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(sd, str(out / "backbone.safetensors"))
        except ImportError:
            raise ImportError("safetensors required. Install with: pip install safetensors")
    else:
        torch.save(sd, out / "backbone.pth")

    logger.info("Exported hub checkpoint to %s", out)
    return out

"""Publish DINO-X models to HuggingFace Hub.

Orchestrates the full publishing pipeline:
1. Load training checkpoint → extract backbone
2. Export hub checkpoint (safetensors + config.json)
3. Generate model card (README.md)
4. Upload to HuggingFace Hub

Requires: huggingface_hub package and a valid HF token.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _scrub_config(config: dict) -> dict:
    """Remove local filesystem paths and hardware details from config."""
    scrubbed = {}
    skip_keys = {"run_dir", "index_csv", "split_manifest"}
    for k, v in config.items():
        if k in skip_keys:
            continue
        if k == "hardware":
            scrubbed[k] = {
                "device_type": v.get("device_type", "unknown"),
                "device_name": v.get("device_name", "unknown"),
            }
            continue
        if isinstance(v, str) and ("/" in v and any(
            p in v for p in ("/home/", "/tmp/", "/workspace/", "/mnt/", "/root/")
        )):
            continue
        scrubbed[k] = v
    return scrubbed


def _scrub_eval(eval_results: dict) -> dict:
    """Remove local paths from eval results."""
    scrubbed = dict(eval_results)
    for key in ("checkpoint",):
        if key in scrubbed and isinstance(scrubbed[key], str):
            parts = scrubbed[key].split("/")
            scrubbed[key] = parts[-1]
    return scrubbed


def publish_to_hub(
    training_checkpoint: str | Path,
    repo_id: str,
    *,
    eval_results_path: str | Path | None = None,
    lineage: dict[str, Any] | None = None,
    token: str | None = None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Publish a trained DINO-X model to HuggingFace Hub.

    Args:
        training_checkpoint: Path to the training .pth checkpoint.
        repo_id: HuggingFace repo ID (e.g., "timlawrenz/dinox-ct-vit-small-v1").
        eval_results_path: Path to evaluation JSON from evaluate_panorgan.py.
        lineage: Dataset provenance dict.
        token: HuggingFace API token. If None, uses HF_TOKEN env var or cached login.
        private: If True, create a private repo.
        dry_run: If True, prepare staging directory but don't upload.

    Returns:
        URL of the published repository (or staging directory path if dry_run).
    """
    import torch
    from huggingface_hub import HfApi

    from zoo.card import generate_model_card
    from zoo.hub import export_hub_checkpoint, load_from_training_checkpoint

    training_checkpoint = Path(training_checkpoint)
    if not training_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {training_checkpoint}")

    # Load training config from checkpoint
    payload = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    training_config = payload.get("config", {})
    model_config = dict(training_config.get("model", {}))
    # Propagate top-level keys into model_config
    for k in ("img_size", "scale_aware"):
        if k in training_config:
            model_config[k] = training_config[k]

    # Count params
    backbone = load_from_training_checkpoint(training_checkpoint)
    n_params = sum(p.numel() for p in backbone.parameters())
    model_config["params_millions"] = round(n_params / 1e6, 1)

    # Load eval results
    eval_results = None
    if eval_results_path:
        eval_results_path = Path(eval_results_path)
        if eval_results_path.exists():
            with open(eval_results_path) as f:
                eval_results = json.load(f)

    # Create staging directory
    staging = Path(tempfile.mkdtemp(prefix="dinox-hub-"))
    logger.info("Staging directory: %s", staging)

    try:
        # Export backbone (safetensors + config.json)
        export_hub_checkpoint(
            backbone=backbone,
            output_dir=str(staging),
            use_safetensors=True,
        )
        logger.info("Exported backbone to %s", staging)

        # Also save as .pth for backward compatibility
        export_hub_checkpoint(
            backbone=backbone,
            output_dir=str(staging),
            use_safetensors=False,
        )

        # Generate and write model card
        card_md = generate_model_card(
            model_config=model_config,
            training_config=_scrub_config(training_config),
            eval_results=eval_results,
            lineage=lineage,
            model_name=repo_id.split("/")[-1] if "/" in repo_id else repo_id,
        )
        (staging / "README.md").write_text(card_md)

        # Write scrubbed training config
        scrubbed_tc = _scrub_config(training_config)
        with open(staging / "training_config.json", "w") as f:
            json.dump(scrubbed_tc, f, indent=2, default=str)

        # Write eval results (scrubbed)
        if eval_results:
            with open(staging / "eval_results.json", "w") as f:
                json.dump(_scrub_eval(eval_results), f, indent=2, default=str)

        # List staging contents
        for p in sorted(staging.iterdir()):
            sz = p.stat().st_size
            logger.info("  %s: %.1f KB", p.name, sz / 1024)

        if dry_run:
            logger.info("Dry run — staging directory: %s", staging)
            return str(staging)

        # Upload to HuggingFace Hub
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            commit_message=f"Upload {repo_id} model + card + eval",
        )

        url = f"https://huggingface.co/{repo_id}"
        logger.info("Published to %s", url)
        return url

    finally:
        if not dry_run:
            shutil.rmtree(staging, ignore_errors=True)

"""Model card generator for DINO-X HuggingFace Hub releases.

Generates standardized model cards with provenance, evaluation results,
and usage examples. Output is HuggingFace-compatible markdown with YAML
frontmatter.

Pure function: dicts in → markdown string out. No I/O.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any


def _yaml_frontmatter(model_config: dict, training_config: dict | None) -> str:
    """Build HuggingFace YAML frontmatter."""
    tags = ["medical-imaging", "vision-transformer", "self-supervised", "dino"]
    if model_config.get("scale_aware"):
        tags.append("scale-aware")

    datasets_used = []
    if training_config:
        for ds in training_config.get("datasets", []):
            if isinstance(ds, str):
                datasets_used.append(ds)
            elif isinstance(ds, dict) and "name" in ds:
                datasets_used.append(ds["name"])

    lines = [
        "---",
        "library_name: dinox",
        "license: cc-by-nc-3.0",
        f"tags: [{', '.join(tags)}]",
        "pipeline_tag: feature-extraction",
    ]
    if datasets_used:
        lines.append(f"datasets: [{', '.join(datasets_used)}]")

    lines.append("---")
    return "\n".join(lines)


def _architecture_section(model_config: dict) -> str:
    """Architecture details table."""
    cfg = model_config
    params_m = cfg.get("params_millions", "—")
    scale = "✅" if cfg.get("scale_aware") else "❌"

    return textwrap.dedent(f"""\
    ## Architecture

    | Parameter | Value |
    |-----------|-------|
    | Backbone | Vision Transformer (ViT) |
    | Config | {cfg.get('name', 'custom')} |
    | Embedding dim | {cfg.get('dim', '—')} |
    | Depth (layers) | {cfg.get('depth', '—')} |
    | Attention heads | {cfg.get('heads', '—')} |
    | Patch size | {cfg.get('patch', '—')} |
    | Image size | {cfg.get('img_size', 224)} |
    | MLP ratio | {cfg.get('mlp_ratio', 4.0)} |
    | Scale-aware | {scale} |
    | Parameters | {params_m}M |
    """)


def _training_section(training_config: dict) -> str:
    """Training configuration details."""
    tc = training_config
    eff_batch = tc.get("batch_size", "?") * tc.get("accumulation_steps", 1)

    return textwrap.dedent(f"""\
    ## Training

    | Parameter | Value |
    |-----------|-------|
    | Method | DINOv3 (self-supervised student-teacher distillation) |
    | Loss | DINO + Gram({tc.get('gram_weight', 1.0)}) + KoLeo({tc.get('koleo_weight', 0.1)}) |
    | Optimizer | AdamW |
    | Learning rate | {tc.get('lr', '—')} (cosine decay to {tc.get('min_lr', '1e-6')}) |
    | Warmup steps | {tc.get('warmup_steps', '—')} |
    | Total steps | {tc.get('max_steps', '—')} |
    | Effective batch | {eff_batch} (batch={tc.get('batch_size', '?')} × accum={tc.get('accumulation_steps', '?')}) |
    | EMA momentum | {tc.get('ema', '—')} |
    | Center momentum | {tc.get('center_momentum', '—')} |
    | Weight decay | {tc.get('weight_decay', '—')} |
    | Seed | {tc.get('train_seed', '—')} |
    | Git commit | `{tc.get('git_commit', '—')}` |
    """)


def _dataset_section(
    training_config: dict | None,
    lineage: dict | None,
) -> str:
    """Dataset provenance and lineage."""
    lines = ["## Training Data\n"]

    if lineage and "datasets" in lineage:
        lines.append("| Dataset | Organ | Slices | Pixel Spacing | Slice Thickness | License |")
        lines.append("|---------|-------|--------|---------------|-----------------|---------|")
        for ds in lineage["datasets"]:
            lines.append(
                f"| {ds.get('name', '—')} | {ds.get('organ', '—')} "
                f"| {ds.get('slices', '—')} | {ds.get('spacing_range', '—')} "
                f"| {ds.get('thickness_range', '—')} | {ds.get('license', '—')} |"
            )
        lines.append("")
    elif training_config:
        idx = training_config.get("index_csv", "—")
        lines.append(f"Training index: `{_scrub_path(idx)}`\n")

    if training_config and "data_manifest_hash" in training_config:
        lines.append(f"Data manifest hash: `{training_config['data_manifest_hash']}`\n")

    return "\n".join(lines)


def _eval_section(eval_results: dict) -> str:
    """Key evaluation metrics table."""
    m = eval_results.get("metrics", {})
    lines = ["## Evaluation\n"]

    # View retrieval
    vr = m.get("view_retrieval_per_dataset", {})
    if vr:
        lines.append("### View Retrieval (self-supervised)\n")
        lines.append("| Dataset | Top-1 | Top-5 | Ratio vs Random |")
        lines.append("|---------|-------|-------|-----------------|")
        for ds_name, vals in vr.items():
            lines.append(
                f"| {ds_name} | {vals.get('top1', 0):.3%} "
                f"| {vals.get('top5', 0):.3%} "
                f"| **{vals.get('ratio_vs_random', 0):.0f}×** |"
            )
        lines.append("")

    # Dataset discrimination
    dd = m.get("dataset_discrimination_probe", {})
    if dd:
        lines.append("### Dataset Discrimination\n")
        lines.append(f"- **Accuracy:** {dd.get('accuracy', 0):.3f}")
        lines.append(f"- **AUC:** {dd.get('auc', 0):.3f}\n")

    # Spacing counterfactual
    sc = m.get("spacing_counterfactual", {})
    if sc:
        d_2x = sc.get("cosine_distance_real_vs_2x", {})
        d_half = sc.get("cosine_distance_real_vs_half", {})
        lines.append("### Scale Awareness (Spacing Counterfactual)\n")
        lines.append(f"- **Real → 2× spacing distance:** {d_2x.get('mean', 0):.4f}")
        lines.append(f"- **Real → ½× spacing distance:** {d_half.get('mean', 0):.4f}")
        lines.append(
            "\nHigher distances = model encodes physical scale "
            "(baseline would be ~0).\n"
        )

    # Spacing prediction
    sp = m.get("spacing_prediction", {})
    if sp:
        lines.append(f"### Spacing Prediction R²: **{sp.get('r2', 0):.3f}**\n")

    lines.append(
        f"*Evaluation on {eval_results.get('val_slices', '?')} validation slices, "
        f"step {eval_results.get('step', '?')}, seed {eval_results.get('seed', '?')}.*\n"
    )

    return "\n".join(lines)


def _usage_section(model_name: str, scale_aware: bool) -> str:
    """Python usage example."""
    spacing_arg = ""
    spacing_import = ""
    if scale_aware:
        spacing_arg = ', spacing=(0.7, 0.7, 1.5)'
        spacing_import = "  # spacing = (pixel_spacing_x, pixel_spacing_y, slice_thickness) in mm\n"

    return textwrap.dedent(f"""\
    ## Usage

    ```python
    from zoo.hub import load_model
    from zoo.encode import encode
    import numpy as np

    # Load pre-trained backbone
    model = load_model("{model_name}")
    model.eval()

    # Encode a CT slice (raw HU values + physical spacing)
    hu_array = np.random.randn(512, 512).astype(np.float32)  # replace with real data
    {spacing_import.rstrip()}
    features = encode(model, hu_array{spacing_arg})
    # features shape: (1, dim) — use for downstream tasks
    ```

    ### Zero-preprocessing API

    The `encode()` function handles windowing, normalization, and resizing
    internally. Pass raw Hounsfield Unit values directly from your PACS system.

    Supported input formats:
    - `hu_float`: Raw HU as float32 numpy array (default)
    - `hu16_png`: 16-bit PNG (offset HU, as produced by our preprocessing)
    - `windowed_float`: Pre-windowed [0, 1] float array

    ### LoRA Fine-Tuning

    ```python
    from zoo.peft import apply_lora, save_adapter

    model = load_model("{model_name}")
    model = apply_lora(model, rank=8)  # ~0.5MB trainable adapter

    # Train on your downstream task...
    # save_adapter(model, "my-adapter/")
    ```
    """)


def _scrub_path(path: str) -> str:
    """Remove local filesystem paths, keep relative parts."""
    # Remove common local prefixes
    for prefix in ("/home/", "/tmp/", "/workspace/", "/root/", "/mnt/"):
        idx = path.find(prefix)
        if idx >= 0:
            # Try to find a meaningful relative part
            parts = path.split("/")
            # Keep from 'data/' or 'runs/' onward
            for i, p in enumerate(parts):
                if p in ("data", "runs", "checkpoints", "experiment"):
                    return "/".join(parts[i:])
            return parts[-1]
    return path


def generate_model_card(
    model_config: dict[str, Any],
    *,
    training_config: dict[str, Any] | None = None,
    eval_results: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
    model_name: str = "dinox-ct-vit-small",
) -> str:
    """Generate a HuggingFace-compatible model card (markdown with YAML frontmatter).

    Args:
        model_config: Model architecture config (dim, depth, heads, patch, etc.).
        training_config: Training hyperparameters and metadata.
        eval_results: Output of evaluate_panorgan.py.
        lineage: Dataset provenance information.
        model_name: Display name / HuggingFace repo ID.

    Returns:
        Markdown string suitable for README.md on HuggingFace Hub.
    """
    sections = []

    # YAML frontmatter
    sections.append(_yaml_frontmatter(model_config, training_config))
    sections.append("")

    # Title and description
    scale_str = "Scale-Aware " if model_config.get("scale_aware") else ""
    sections.append(f"# {model_name}\n")
    sections.append(
        f"A {scale_str}Vision Foundation Model for volumetric medical imaging, "
        f"trained with DINOv3 self-supervised learning on multi-organ CT data.\n"
    )
    sections.append(
        "Part of the [DINO-X model zoo](https://github.com/timlawrenz/DINO-X) — "
        "open-source, pan-organ, scale-aware foundation models for medical imaging.\n"
    )

    # Disclaimer
    sections.append(
        "> ⚠️ **Research use only.** This model is not approved for clinical "
        "diagnosis or treatment decisions. It has not been validated by regulatory "
        "bodies (FDA, CE, etc.). Always consult qualified medical professionals.\n"
    )

    # Architecture
    sections.append(_architecture_section(model_config))

    # Training
    if training_config:
        sections.append(_training_section(training_config))

    # Data
    if training_config or lineage:
        sections.append(_dataset_section(training_config, lineage))

    # Evaluation
    if eval_results:
        sections.append(_eval_section(eval_results))

    # Usage
    sections.append(_usage_section(model_name, model_config.get("scale_aware", False)))

    # Citation
    sections.append(textwrap.dedent("""\
    ## Citation

    ```bibtex
    @software{dinox2026,
      author = {Lawrenz, Tim},
      title = {DINO-X: Scale-Aware Vision Foundation Models for Medical Imaging},
      year = {2026},
      url = {https://github.com/timlawrenz/DINO-X}
    }
    ```
    """))

    # License
    sections.append(textwrap.dedent("""\
    ## License

    Model weights are released under **CC-BY-NC-3.0** (non-commercial), consistent
    with the most restrictive upstream dataset license (LIDC-IDRI).

    Training code is released under **GPL-3.0**.
    """))

    # Generation timestamp
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sections.append(f"---\n*Model card auto-generated by DINO-X zoo v0.1 at {ts}*\n")

    return "\n".join(sections)

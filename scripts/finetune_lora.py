#!/usr/bin/env python3
"""LoRA fine-tuning script for DINO-X backbones.

Downloads a pretrained backbone (from HuggingFace or local checkpoint),
injects LoRA adapters, adds a task head, and runs standard supervised
training on labeled data.

Only the LoRA adapter weights (~0.5-5 MB) and the task head are saved,
keeping the backbone frozen. This enables hospital researchers with
limited hardware to fine-tune massive backbones for specific tasks.

Usage::

    python scripts/finetune_lora.py \\
      --backbone runs/v1-local-5k/hub-export/ \\
      --train-csv data/train_labels.csv \\
      --val-csv data/val_labels.csv \\
      --task classification --num-classes 5 \\
      --rank 8 --epochs 50 --lr 1e-3 \\
      --output adapters/my-adapter/

CSV format:
    image_path,label[,spacing_x,spacing_y,spacing_z]

    - image_path: path to image file (absolute or relative to --data-root)
    - label: integer for classification, float for regression
    - spacing_x/y/z: optional physical spacing in mm (from DICOM header)

Note: This script performs **slice-level** fine-tuning on single images.
The pretrained backbone was trained with 3-slice (z-1, z, z+1) context;
single-image input is a distribution shift. For the best results, provide
3-channel images representing consecutive slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Ensure repo root is on sys.path for zoo imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zoo.arch import PatchViT
from zoo.hub import load_model
from zoo.peft import apply_lora, count_parameters, save_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LabeledRow:
    image_path: Path
    label: float
    spacing_x: float = 1.0
    spacing_y: float = 1.0
    spacing_z: float = 1.0
    has_spacing: bool = False


@dataclass
class FinetuneConfig:
    """Metadata saved alongside the adapter for reproducibility."""
    backbone: str
    task: str
    num_classes: int
    rank: int
    alpha: float
    lr: float
    epochs: int
    batch_size: int
    input_format: str
    scale_aware: bool
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_metrics: dict[str, float] = field(default_factory=dict)
    train_samples: int = 0
    val_samples: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[LabeledRow]:
    """Parse a labeled CSV file into LabeledRow list."""
    rows: list[LabeledRow] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")

        required = {"image_path", "label"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {path} missing required columns: {missing}. "
                f"Found: {reader.fieldnames}"
            )

        has_spacing = all(
            c in reader.fieldnames for c in ("spacing_x", "spacing_y", "spacing_z")
        )

        for i, row in enumerate(reader):
            try:
                rows.append(LabeledRow(
                    image_path=Path(row["image_path"]),
                    label=float(row["label"]),
                    spacing_x=float(row["spacing_x"]) if has_spacing else 1.0,
                    spacing_y=float(row["spacing_y"]) if has_spacing else 1.0,
                    spacing_z=float(row["spacing_z"]) if has_spacing else 1.0,
                    has_spacing=has_spacing,
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Error parsing row {i + 1} of {path}: {e}") from e

    return rows


class LabeledImageDataset(Dataset):
    """Single-image dataset for fine-tuning with optional spacing."""

    def __init__(
        self,
        rows: list[LabeledRow],
        img_size: int = 224,
        input_format: str = "hu16_png",
        window_level: float = 40.0,
        window_width: float = 400.0,
        augment: bool = False,
        data_root: Path | None = None,
    ):
        self.rows = rows
        self.img_size = img_size
        self.input_format = input_format
        self.window_level = window_level
        self.window_width = window_width
        self.data_root = data_root

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    img_size, scale=(0.7, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    img_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(img_size),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, p: Path) -> Path:
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    def _load_image(self, p: Path) -> np.ndarray:
        """Load image and convert to [0, 1] float array of shape (H, W)."""
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)

        # Handle multi-channel images
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        if self.input_format == "hu16_png":
            hu = (arr - 32768.0) * 0.1
            wmin = self.window_level - self.window_width / 2.0
            windowed = (hu - wmin) / max(self.window_width, 1.0)
            return np.clip(windowed, 0.0, 1.0)
        elif self.input_format == "float01":
            return np.clip(arr / 255.0 if arr.max() > 1.0 else arr, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        row = self.rows[idx]
        p = self._resolve_path(row.image_path)

        arr = self._load_image(p)
        # Replicate single channel to 3 channels (matches pretraining)
        x = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
        x = self.transform(torch.from_numpy(x).contiguous())

        spacing = torch.tensor(
            [row.spacing_x, row.spacing_y, row.spacing_z],
            dtype=torch.float32,
        )

        return x, spacing, row.label


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class FinetuneModel(nn.Module):
    """Backbone (with LoRA) + task head.

    The head is kept outside of PEFT for clean save/load: adapter weights
    are saved via peft, head weights are saved separately.
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        num_classes: int,
        task: str = "classification",
    ):
        super().__init__()
        self.backbone = backbone
        self.task = task

        if task == "regression":
            self.head = nn.Linear(dim, 1)
        else:
            self.head = nn.Linear(dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.backbone(x, spacing=spacing)
        cls_token = features[:, 0]  # (B, dim)
        return self.head(cls_token)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _compute_auroc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute AUROC for binary classification (no sklearn dependency).

    Uses the rank-based formula: AUROC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    """
    probs_np = probs.numpy()
    targets_np = targets.numpy()

    pos_mask = targets_np == 1
    n_pos = pos_mask.sum()
    n_neg = len(targets_np) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined, return chance

    # Rank-based AUROC (handles ties correctly)
    order = np.argsort(probs_np)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs_np) + 1, dtype=np.float64)

    # Handle ties: average ranks for tied values
    sorted_probs = probs_np[order]
    i = 0
    while i < len(sorted_probs):
        j = i + 1
        while j < len(sorted_probs) and sorted_probs[j] == sorted_probs[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    sum_pos_ranks = ranks[pos_mask].sum()
    auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    num_classes: int,
) -> dict[str, float]:
    """Compute metrics for classification or regression."""
    metrics: dict[str, float] = {}

    if task == "classification":
        pred_labels = preds.argmax(dim=-1)
        correct = (pred_labels == targets).sum().item()
        total = targets.size(0)
        metrics["accuracy"] = correct / total if total > 0 else 0.0

        # AUROC for binary classification
        if num_classes == 2:
            # Use softmax probability of the positive class
            probs = torch.softmax(preds, dim=-1)[:, 1]
            metrics["auroc"] = _compute_auroc(probs, targets)

        # Per-class accuracy and macro-F1
        if num_classes > 1:
            per_class_tp = torch.zeros(num_classes)
            per_class_fp = torch.zeros(num_classes)
            per_class_fn = torch.zeros(num_classes)
            for c in range(num_classes):
                tp = ((pred_labels == c) & (targets == c)).sum().float()
                fp = ((pred_labels == c) & (targets != c)).sum().float()
                fn = ((pred_labels != c) & (targets == c)).sum().float()
                per_class_tp[c] = tp
                per_class_fp[c] = fp
                per_class_fn[c] = fn

            precisions = per_class_tp / (per_class_tp + per_class_fp + 1e-8)
            recalls = per_class_tp / (per_class_tp + per_class_fn + 1e-8)
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)

            # Only average over classes that appear in targets
            present = torch.zeros(num_classes, dtype=torch.bool)
            for c in range(num_classes):
                present[c] = (targets == c).any()

            if present.any():
                metrics["macro_f1"] = f1s[present].mean().item()
            else:
                metrics["macro_f1"] = 0.0

    elif task == "regression":
        preds_flat = preds.squeeze(-1)
        mse = F.mse_loss(preds_flat, targets).item()
        metrics["mse"] = mse
        metrics["rmse"] = math.sqrt(mse)

        # R² (handle constant target)
        ss_res = ((preds_flat - targets) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        metrics["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

    return metrics


def _train_one_epoch(
    model: FinetuneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    task: str,
    scale_aware: bool,
    amp_enabled: bool,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, spacings, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spacing = spacings.to(device) if scale_aware else None

        if task == "classification":
            labels = labels.long()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images, spacing=spacing)
            if task == "classification":
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.mse_loss(logits.squeeze(-1), labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: FinetuneModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
    num_classes: int,
    scale_aware: bool,
    amp_enabled: bool,
) -> tuple[float, dict[str, float]]:
    """Validate and return (loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for images, spacings, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        spacing = spacings.to(device) if scale_aware else None

        if task == "classification":
            labels = labels.long()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images, spacing=spacing)
            if task == "classification":
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.mse_loss(logits.squeeze(-1), labels)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(preds, targets, task, num_classes)
    metrics["loss"] = avg_loss

    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Save / load adapter + head
# ---------------------------------------------------------------------------


def save_finetune(
    model: FinetuneModel,
    output_dir: Path,
    config: FinetuneConfig,
) -> None:
    """Save adapter weights, task head, and config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter via peft
    save_adapter(model.backbone, output_dir)

    # Save task head separately
    torch.save(model.head.state_dict(), output_dir / "head.pth")

    # Save metadata
    config_dict = asdict(config)
    (output_dir / "finetune_config.json").write_text(
        json.dumps(config_dict, indent=2, default=str)
    )

    logger.info("Saved fine-tuned adapter + head to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a DINO-X backbone with LoRA adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--backbone", required=True,
        help="Backbone source: HuggingFace model ID, local .pth, or hub directory",
    )
    parser.add_argument("--device", default="cuda", help="Device for training")

    # Data
    parser.add_argument("--train-csv", required=True, type=Path, help="Training CSV")
    parser.add_argument("--val-csv", required=True, type=Path, help="Validation CSV")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--input-format", default="hu16_png",
                        choices=["hu16_png", "float01"],
                        help="Image encoding format")
    parser.add_argument("--window-level", type=float, default=40.0,
                        help="HU window center (hu16_png only)")
    parser.add_argument("--window-width", type=float, default=400.0,
                        help="HU window width (hu16_png only)")

    # Task
    parser.add_argument("--task", default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes (classification only)")

    # LoRA
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--es-metric", default="loss",
                        help="Early stopping metric: 'loss' (lower is better) or "
                             "any metric name like 'auroc' (higher is better)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")

    # Output
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to save adapter + head")

    args = parser.parse_args(argv)

    device = torch.device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"

    # ── Load backbone ──────────────────────────────────────────────────
    logger.info("Loading backbone: %s", args.backbone)
    backbone = load_model(args.backbone, device=str(device))
    scale_aware = backbone.scale_aware
    dim = backbone.dim
    img_size = backbone.img_size

    logger.info(
        "Backbone: dim=%d, depth=%d, scale_aware=%s, img_size=%d",
        dim, len(backbone.blocks), scale_aware, img_size,
    )

    # ── Load data ──────────────────────────────────────────────────────
    train_rows = _load_csv(args.train_csv)
    val_rows = _load_csv(args.val_csv)
    logger.info("Train: %d samples, Val: %d samples", len(train_rows), len(val_rows))

    # Spacing warning for scale-aware backbones
    if scale_aware and not train_rows[0].has_spacing:
        logger.warning(
            "⚠️  Scale-aware backbone but CSV has no spacing columns! "
            "Using default (1,1,1) — this may reduce fine-tuning quality. "
            "Add spacing_x, spacing_y, spacing_z columns for best results."
        )

    # Validate classification labels
    if args.task == "classification":
        all_labels = set(int(r.label) for r in train_rows + val_rows)
        max_label = max(all_labels)
        if max_label >= args.num_classes:
            parser.error(
                f"Found label {max_label} but --num-classes={args.num_classes}. "
                f"Labels must be in [0, {args.num_classes - 1}]."
            )
        missing = set(range(args.num_classes)) - all_labels
        if missing:
            logger.warning(
                "Classes %s have no samples in train+val. "
                "Consider reducing --num-classes.",
                sorted(missing),
            )

    train_ds = LabeledImageDataset(
        train_rows, img_size=img_size, input_format=args.input_format,
        window_level=args.window_level, window_width=args.window_width,
        augment=True, data_root=args.data_root,
    )
    val_ds = LabeledImageDataset(
        val_rows, img_size=img_size, input_format=args.input_format,
        window_level=args.window_level, window_width=args.window_width,
        augment=False, data_root=args.data_root,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Build model ────────────────────────────────────────────────────
    # Apply LoRA (backbone is still in eval mode from load_model)
    backbone = apply_lora(
        backbone, rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout,
    )

    model = FinetuneModel(
        backbone=backbone,
        dim=dim,
        num_classes=args.num_classes,
        task=args.task,
    ).to(device)

    # Count parameters
    params = count_parameters(backbone)
    head_params = sum(p.numel() for p in model.head.parameters())
    logger.info(
        "Parameters — backbone: %d total (%d trainable LoRA), head: %d",
        params["total"], params["trainable"], head_params,
    )

    # ── Optimizer + scheduler ──────────────────────────────────────────
    trainable_params = [
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, weight_decay=args.weight_decay,
    )

    # Cosine schedule with linear warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # ── Training config for metadata ───────────────────────────────────
    ft_config = FinetuneConfig(
        backbone=args.backbone,
        task=args.task,
        num_classes=args.num_classes,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_format=args.input_format,
        scale_aware=scale_aware,
        train_samples=len(train_rows),
        val_samples=len(val_rows),
    )

    # ── Training loop ──────────────────────────────────────────────────
    # Early stopping: higher-is-better metrics vs lower-is-better
    es_metric = args.es_metric
    es_higher_is_better = es_metric != "loss"
    best_es_value = float("-inf") if es_higher_is_better else float("inf")
    patience_counter = 0
    start_time = time.time()

    logger.info("Starting training: %d epochs, %d steps/epoch", args.epochs, len(train_loader))
    logger.info("AMP: %s, Device: %s, ES metric: %s", amp_enabled, device, es_metric)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            args.task, scale_aware, amp_enabled,
        )

        val_loss, val_metrics = _validate(
            model, val_loader, device, args.task, args.num_classes,
            scale_aware, amp_enabled,
        )

        epoch_time = time.time() - epoch_start

        # Format metrics string
        metric_parts = [f"val_loss={val_loss:.4f}"]
        for k, v in val_metrics.items():
            if k != "loss":
                metric_parts.append(f"{k}={v:.4f}")

        logger.info(
            "Epoch %d/%d — train_loss=%.4f %s (%.1fs)",
            epoch, args.epochs, train_loss,
            " ".join(metric_parts), epoch_time,
        )

        # Get early stopping metric value
        if es_metric == "loss":
            current_es = val_loss
        elif es_metric in val_metrics:
            current_es = val_metrics[es_metric]
        else:
            logger.warning(
                "ES metric '%s' not found in val_metrics %s, falling back to loss",
                es_metric, list(val_metrics.keys()),
            )
            current_es = val_loss
            es_higher_is_better = False

        improved = (
            current_es > best_es_value if es_higher_is_better
            else current_es < best_es_value
        )

        if improved:
            best_es_value = current_es
            patience_counter = 0

            ft_config.best_epoch = epoch
            ft_config.best_val_loss = val_loss
            ft_config.best_val_metrics = val_metrics

            save_finetune(model, args.output, ft_config)
            logger.info("  ✓ Saved best model (%s=%.4f)", es_metric, current_es)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, args.patience,
                )
                break

    elapsed = time.time() - start_time
    logger.info(
        "Training complete in %.1fs — best epoch %d, val_loss=%.4f",
        elapsed, ft_config.best_epoch, ft_config.best_val_loss,
    )

    # Print final metrics
    if ft_config.best_val_metrics:
        logger.info("Best validation metrics:")
        for k, v in ft_config.best_val_metrics.items():
            logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()

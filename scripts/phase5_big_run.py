#!/usr/bin/env python3
"""Phase 5: Big Run - Production DINOv3 training with parameterized model configurations.

This script supports both Phase 5a (ViT-Large validation) and Phase 5b (ViT-Giant production):
- Parameterized model configuration (vit-large, vit-giant, custom)
- Hardware-agnostic checkpoints (train on 4090, resume on amd395)
- Extended checkpoint management with rotation
- Gradient accumulation for large effective batch sizes
- Automatic hardware detection and optimization
- Training anomaly detection
- Full monitoring integration

Example usage:
    # Phase 5a: ViT-Large validation run (384 steps)
    python scripts/phase5_big_run.py --config vit-large --max-steps 384 --num-workers 8

    # Phase 5b: ViT-Giant production run (unlimited)
    python scripts/phase5_big_run.py --config vit-giant

    # Resume from checkpoint (auto-detects latest)
    python scripts/phase5_big_run.py --config vit-large --resume auto

    # Cross-hardware resume (4090 → amd395)
    python scripts/phase5_big_run.py --config vit-large --resume path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `zoo` package is importable
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Optional import helpers
def _need(mod: str) -> None:
    raise SystemExit(
        f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry. "
        "If you're using ROCm PyTorch, ensure ROCm libs are discoverable (e.g., `source scripts/rocm_env.*`)."
    )

try:
    import numpy as np
except Exception:
    _need("numpy")

try:
    from PIL import Image
except Exception:
    _need("pillow")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
except Exception as e:
    _need(f"torch ({e})")

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # tensorboard is optional


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_attention_heatmap(
    model: DinoStudentTeacher,
    batch: torch.Tensor,
) -> np.ndarray:
    """Generate attention heatmap for the first image in batch."""
    model.eval()
    with torch.no_grad():
        feats = model.backbone(batch[0:1]) # (1, N+1, D)
    model.train()
    
    # L2 norm of patch tokens (skip CLS and skip Registers if present)
    # feats: (1, N_total, D)
    # Patches are at [1 : 1+N_patches]
    num_patches = model.backbone.patch_embed.grid_size[0] * model.backbone.patch_embed.grid_size[1] if hasattr(model.backbone.patch_embed, 'grid_size') else int((model.backbone.img_size // model.backbone.patch)**2)
    
    patch_tokens = feats[:, 1 : 1 + num_patches, :] # (1, N_patches, D)
    norms = torch.norm(patch_tokens, dim=-1).squeeze(0) # (N_patches,)
    
    # Normalize to [0, 1]
    norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    
    # Reshape to grid
    grid_size = int(np.sqrt(norms.shape[0]))
    heatmap = norms.reshape(grid_size, grid_size).cpu().numpy()
    
    # Resize to match input image size for better visualization
    # We return the small heatmap; TensorBoard handles resizing if needed,
    # or we can rely on the viewer. 
    return heatmap


def make_gram_heatmap(
    model: DinoStudentTeacher,
    batch: torch.Tensor,
) -> np.ndarray:
    """Generate Gram matrix visualization for the first image in batch."""
    model.eval()
    with torch.no_grad():
        feats = model.backbone(batch[0:1]) # (1, N+1, D)
    model.train()
    
    # Compute Gram matrix of patch tokens (skip CLS)
    # feats: (1, N+1, D) -> patches: (1, N, D)
    patches = feats[:, 1:, :] 
    
    # Normalize features first (standard Gram practice)
    patches = F.normalize(patches, p=2, dim=-1)
    
    # Gram: (1, N, N)
    gram = torch.bmm(patches, patches.transpose(1, 2)).squeeze(0) # (N, N)
    
    # Convert to numpy
    gram_np = gram.cpu().numpy()
    
    # Contrast Stretching: Map [min, max] to [0, 1]
    g_min = gram_np.min()
    g_max = gram_np.max()
    gram_img = (gram_np - g_min) / (g_max - g_min + 1e-8)
    
    return gram_img




# ─────────────────────────────────────────────────────────────────────────────
# Configuration Presets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    patch: int
    dim: int
    depth: int
    heads: int
    mlp_ratio: float = 4.0
    out_dim: int = 8192

    def __post_init__(self):
        """Validate configuration."""
        if self.dim % self.heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")
        if self.patch not in [8, 14, 16]:
            warnings.warn(f"Unusual patch size: {self.patch}")

    @property
    def params_millions(self) -> float:
        """Rough parameter count estimate (millions)."""
        # Patch embed: 3 * patch^2 * dim
        # Transformer blocks: depth * (4 * dim^2 + 8 * dim * dim * mlp_ratio)
        # Projection head: dim * out_dim * 2
        patch_embed = 3 * self.patch * self.patch * self.dim
        transformer = self.depth * (4 * self.dim * self.dim + 8 * self.dim * self.dim * self.mlp_ratio)
        head = self.dim * self.out_dim * 2
        return (patch_embed + transformer + head) / 1e6


# Model configuration presets
MODEL_CONFIGS = {
    "vit-tiny": ModelConfig(
        name="vit-tiny",
        patch=14,
        dim=192,
        depth=12,
        heads=3,
        mlp_ratio=4.0,
        out_dim=4096,
    ),
    "vit-small": ModelConfig(
        name="vit-small",
        patch=14,
        dim=384,
        depth=12,
        heads=6,
        mlp_ratio=4.0,
        out_dim=8192,
    ),
    "vit-large": ModelConfig(
        name="vit-large",
        patch=14,
        dim=1024,
        depth=24,
        heads=16,
        mlp_ratio=4.0,
        out_dim=8192,
    ),
    "vit-giant": ModelConfig(
        name="vit-giant",
        patch=14,
        dim=1408,
        depth=40,
        heads=16,
        mlp_ratio=4.0,
        out_dim=8192,
    ),
}


@dataclass
class HardwareConfig:
    """Hardware-specific optimization settings."""
    device_type: str  # "cuda" or "cpu"
    device_name: str
    is_rocm: bool
    num_workers: int
    pin_memory: bool
    batch_size_recommendation: int


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Model
    model: ModelConfig
    img_size: int = 224
    
    # Hardware
    hardware: HardwareConfig | None = None
    
    # Data augmentation
    rw_level_min: float = -400.0
    rw_level_max: float = 400.0
    rw_width_min: float = 800.0
    rw_width_max: float = 2000.0
    
    # Training
    batch_size: int = 64
    accumulation_steps: int = 1
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 2500
    weight_decay: float = 0.04
    max_steps: int | None = None
    
    # DINO
    ema: float = 0.996
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    center_momentum: float = 0.9
    
    loss_type: str = "dino"
    
    # Gram anchoring - ALWAYS ENABLED (required for medical imaging)
    # Without Gram Anchoring, the model will collapse on CT scans
    gram_enabled: bool = True  # DO NOT CHANGE - hardcoded to True
    gram_weight: float = 1.0
    koleo_weight: float = 0.0
    
    # Scale awareness
    scale_aware: bool = False
    
    # Anti-memorization
    crop_scale_min: float = 0.3
    crop_scale_max: float = 1.0
    z_stride: int = 1
    diverse_batches: bool = False
    
    # Checkpointing
    ckpt_every: int = 100
    ckpt_keep_last: int = 5
    
    # Monitoring
    monitor_every: int = 1000
    
    # Seeds and reproducibility
    train_seed: int = 0
    sdp_backend: str = "auto"
    amp_dtype: str = "bfloat16"
    
    # Data paths
    index_csv: str = "data/processed/_index/index.csv"
    split_manifest: str | None = None
    
    # Provenance
    git_commit: str | None = None
    data_manifest_hash: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> HardwareConfig:
    """Detect hardware and apply optimizations."""
    if not torch.cuda.is_available():
        return HardwareConfig(
            device_type="cpu",
            device_name="CPU",
            is_rocm=False,
            num_workers=4,
            pin_memory=False,
            batch_size_recommendation=8,
        )
    
    device_name = torch.cuda.get_device_name(0)
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    
    # Hardware-specific optimizations (from Phase 4.5 throughput tuning)
    if "4090" in device_name or "RTX 4090" in device_name:
        return HardwareConfig(
            device_type="cuda",
            device_name=device_name,
            is_rocm=False,
            num_workers=8,
            pin_memory=True,
            batch_size_recommendation=64,
        )
    elif is_rocm and ("gfx1151" in device_name or "Strix" in device_name or "8060S" in device_name):
        # AMD Strix Halo (amd395) - unified memory optimizations
        return HardwareConfig(
            device_type="cuda",
            device_name=device_name,
            is_rocm=True,
            num_workers=16,
            pin_memory=False,  # Unified memory benefits from no pinning
            batch_size_recommendation=128,
        )
    else:
        # Generic CUDA device
        return HardwareConfig(
            device_type="cuda",
            device_name=device_name,
            is_rocm=is_rocm,
            num_workers=4,
            pin_memory=True,
            batch_size_recommendation=32,
        )


def get_git_commit() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        if result.stdout.strip():
            warnings.warn("Uncommitted changes detected. Reproducibility may be affected.")
            commit += "-dirty"
        
        return commit
    except Exception:
        return None


def compute_data_manifest_hash(index_csv: Path) -> str | None:
    """Compute hash of data manifest for provenance."""
    try:
        if not index_csv.exists():
            return None
        
        hasher = hashlib.sha256()
        with open(index_csv, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()[:16]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture — imported from zoo.arch
# ─────────────────────────────────────────────────────────────────────────────

from zoo.arch import (  # noqa: E402
    DinoStudentTeacher,
    PatchViT,
    ScaleEmbedding,
    TransformerBlock,
    migrate_state_dict,
    needs_migration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (from Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexRow:
    png_path: Path
    series_dir: str
    slice_index: int
    encoding: str
    spacing_x: float = 1.0
    spacing_y: float = 1.0
    spacing_z: float = 1.0
    dataset: str = ""


def _load_index_rows(index_csv: Path, require_spacing: bool = False) -> list[IndexRow]:
    """Load index CSV.

    Args:
        index_csv: Path to CSV with columns: png_path, series_dir, slice_index, encoding,
                   and optionally spacing_x, spacing_y, spacing_z, dataset.
        require_spacing: If True, warn when spacing columns are missing.
    """
    rows = []
    with open(index_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_spacing = all(c in fieldnames for c in ("spacing_x", "spacing_y", "spacing_z"))
        has_dataset = "dataset" in fieldnames

        if require_spacing and not has_spacing:
            warnings.warn(
                f"--scale-aware is enabled but {index_csv} lacks spacing_x/spacing_y/spacing_z columns. "
                "Defaulting to (1.0, 1.0, 1.0) — the model won't learn real scale awareness."
            )

        for r in reader:
            kwargs: dict[str, Any] = {
                "png_path": Path(r["png_path"]),
                "series_dir": r["series_dir"],
                "slice_index": int(r["slice_index"]),
                "encoding": r["encoding"],
            }
            if has_spacing:
                kwargs["spacing_x"] = float(r["spacing_x"])
                kwargs["spacing_y"] = float(r["spacing_y"])
                kwargs["spacing_z"] = float(r["spacing_z"])
            if has_dataset:
                kwargs["dataset"] = r["dataset"]
            rows.append(IndexRow(**kwargs))
    return rows


class PngDataset(torch.utils.data.Dataset):
    """Dataset for loading 16-bit HU PNG slices with 3-slice context and random windowing."""
    def __init__(
        self,
        rows: list[IndexRow],
        img_size: int = 224,
        rw_level_min: float = -400.0,
        rw_level_max: float = 400.0,
        rw_width_min: float = 800.0,
        rw_width_max: float = 2000.0,
        scale_aware: bool = False,
        crop_scale_min: float = 0.3,
        crop_scale_max: float = 1.0,
    ):
        self.rows = rows
        self.img_size = img_size
        self.rw_level_min = rw_level_min
        self.rw_level_max = rw_level_max
        self.rw_width_min = rw_width_min
        self.rw_width_max = rw_width_max
        self.scale_aware = scale_aware

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(crop_scale_min, crop_scale_max), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Map series for 3-slice context (z-1, z, z+1)
        self._series_map: dict[str, dict[int, Path]] = {}
        self._series_minmax: dict[str, tuple[int, int]] = {}
        for r in rows:
            sm = self._series_map.setdefault(r.series_dir, {})
            sm[r.slice_index] = r.png_path
        for s, mp in self._series_map.items():
            if mp:
                ks = sorted(mp.keys())
                self._series_minmax[s] = (ks[0], ks[-1])

    def __len__(self) -> int:
        return len(self.rows)

    def _load_hu01(self, p: Path, level: float, width: float) -> np.ndarray:
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        
        wmin = level - width / 2.0
        wmax = level + width / 2.0
        windowed = (hu - wmin) / max(width, 1.0)
        windowed = np.clip(windowed, 0.0, 1.0)
            
        return windowed

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Robust data loading loop
        attempts = 0
        while attempts < 10:
            try:
                row = self.rows[idx]
                s = row.series_dir
                z = row.slice_index
                z0, z1 = self._series_minmax.get(s, (z, z))

                def _clamp(k: int) -> int:
                    return max(z0, min(z1, k))

                mp = self._series_map.get(s, {})
                paths = [
                    mp.get(_clamp(z - 1), row.png_path),
                    mp.get(_clamp(z), row.png_path),
                    mp.get(_clamp(z + 1), row.png_path),
                ]

                def _get_view():
                    level = random.uniform(self.rw_level_min, self.rw_level_max)
                    width = random.uniform(self.rw_width_min, self.rw_width_max)
                    
                    slices = [self._load_hu01(p, level, width) for p in paths]
                    x = np.stack(slices, axis=0) # (3, H, W)
                    return self.transform(torch.from_numpy(x).contiguous())

                spacing = torch.tensor(
                    [row.spacing_x, row.spacing_y, row.spacing_z],
                    dtype=torch.float32,
                )

                # Return two different windowed views for DINO cross-view prediction
                return [_get_view(), _get_view()], spacing
            
            except Exception as e:
                print(f"⚠️  Data loading error at index {idx} ({self.rows[idx].png_path}): {e}")
                # Pick a new random index
                idx = random.randint(0, len(self.rows) - 1)
                attempts += 1
        
        # Fallback if 10 attempts fail (unlikely)
        raise RuntimeError("Failed to load data after 10 attempts")


class DiverseBatchSampler(torch.utils.data.Sampler):
    """Round-robin batch sampler ensuring at most one sample per series per batch.

    Prevents same-series collisions that create trivially easy contrastive pairs.
    Each epoch sees all samples; they are just rearranged into series-diverse batches.
    """

    def __init__(self, rows: list, batch_size: int, drop_last: bool = True,
                 generator: torch.Generator | None = None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

        # Group indices by series_dir
        self._series_indices: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            self._series_indices.setdefault(row.series_dir, []).append(i)

        self._total = len(rows)

    def __len__(self) -> int:
        if self.drop_last:
            return self._total // self.batch_size
        return (self._total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Shuffle indices within each series
        queues: list[list[int]] = []
        for indices in self._series_indices.values():
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            queues.append([indices[i] for i in perm])

        # Shuffle series order
        series_order = torch.randperm(len(queues), generator=self.generator).tolist()
        queues = [queues[i] for i in series_order]

        # Interleave: round-robin one sample from each series
        interleaved: list[int] = []
        while queues:
            next_round: list[list[int]] = []
            for q in queues:
                interleaved.append(q.pop())
                if q:
                    next_round.append(q)
            queues = next_round

        # Chunk into batches
        for i in range(0, len(interleaved) - self.batch_size + 1, self.batch_size):
            yield interleaved[i : i + self.batch_size]

        # Final partial batch
        remainder = interleaved[len(interleaved) // self.batch_size * self.batch_size :]
        if remainder and not self.drop_last:
            yield remainder


def dino_collate(
    batch: list[tuple[list[torch.Tensor], torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Custom collate for PngDataset that returns (views, spacing).

    Args:
        batch: List of (views_list, spacing_tensor) from PngDataset.__getitem__

    Returns:
        ([view1_batch, view2_batch], spacing_batch) where:
        - view1_batch, view2_batch: (B, 3, H, W)
        - spacing_batch: (B, 3)
    """
    views_lists, spacings = zip(*batch)
    v1 = torch.stack([v[0] for v in views_lists])
    v2 = torch.stack([v[1] for v in views_lists])
    spacing_batch = torch.stack(list(spacings))
    return [v1, v2], spacing_batch


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    # Linear warmup
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    # If total_steps is not defined (unlimited run), just return base_lr after warmup
    if total_steps is None:
        return base_lr
    
    # If we are past total_steps, return min_lr
    if step >= total_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


class DINOLoss(nn.Module):
    """DINO loss with centering and sharpening to prevent collapse."""
    def __init__(self, out_dim: int, center_momentum: float = 0.999) -> None:
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update exponential moving average of teacher outputs."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        student_temp: float,
        teacher_temp: float,
    ) -> torch.Tensor:
        """Compute DINO cross-entropy loss with cross-view prediction."""
        # teacher_out is already no_grad from the loop
        
        # teacher sharpening and centering
        teacher_prob = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        
        # student log-probabilities
        student_log_prob = F.log_softmax(student_out / student_temp, dim=-1)
        
        # Cross-view prediction:
        # student_out and teacher_out contain [batch_v1, batch_v2]
        # We want H(teacher_v1, student_v2) and H(teacher_v2, student_v1)
        B = teacher_out.shape[0] // 2
        t1, t2 = teacher_prob[:B], teacher_prob[B:]
        s1, s2 = student_log_prob[:B], student_log_prob[B:]
        
        loss1 = -torch.sum(t1 * s2, dim=-1).mean()
        loss2 = -torch.sum(t2 * s1, dim=-1).mean()
        loss = (loss1 + loss2) / 2.0
        
        self.update_center(teacher_out)
        return loss


def compute_gram_matrix(feats: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for patch tokens."""
    B, N, D = feats.shape
    feats_norm = F.normalize(feats, p=2, dim=-1)
    gram = torch.bmm(feats_norm, feats_norm.transpose(1, 2))
    return gram


def compute_gram_anchoring_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """Gram Anchoring loss to preserve texture correlations."""
    student_gram = compute_gram_matrix(student_feats[:, 1:])  # Skip CLS token
    teacher_gram = compute_gram_matrix(teacher_feats[:, 1:])
    loss = F.mse_loss(student_gram, teacher_gram)
    return loss


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko differential entropy regularization.
    
    Forces features to be uniformly distributed on the hypersphere, preventing collapse.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        student_output: (B, D) tensor of features (CLS tokens).
        """
        # Normalize features
        x = F.normalize(student_output, p=2, dim=-1)
        
        # Compute pairwise distances
        # dist(i, j) = 2 - 2 * cos(i, j)
        # We can just use pdist
        pdist = torch.cdist(x, x, p=2) # (B, B)
        
        # Mask diagonal (distance to self is 0)
        B = x.shape[0]
        # Create a mask with infinity on diagonal
        eye = torch.eye(B, device=x.device)
        pdist = pdist + eye * 1e9
        
        # Find distance to nearest neighbor for each sample
        min_dist, _ = pdist.min(dim=1) # (B,)
        
        # Maximize entropy = Maximize log(min_dist) = Minimize -log(min_dist)
        loss = -torch.log(min_dist + eps).mean()
        return loss


class SimCLRLoss(nn.Module):
    """SimCLR / NT-Xent Loss.
    
    Explicitly uses negative samples from the batch to prevent collapse.
    robust, but requires large batch sizes (which we have via accumulation).
    """
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) feature vectors from view 1 and view 2.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate: (2B, D)
        features = torch.cat([z1, z2], dim=0)
        B = z1.shape[0]
        
        # Similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * B, device=features.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive targets
        # For i in 0..B-1 (z1), positive is i+B (z2)
        # For i in B..2B-1 (z2), positive is i-B (z1)
        target = torch.cat([
            torch.arange(B, 2 * B, device=features.device),
            torch.arange(0, B, device=features.device)
        ], dim=0)
        
        loss = F.cross_entropy(sim_matrix, target)
        return loss


class MaeDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_patches: int,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Fixed 2D sin-cos pos embedding for decoder (recomputed on init)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3, bias=True) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # Embed latent tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixels
        x = self.decoder_pred(x)
        return x[:, 1:, :] # remove cls token


class MaeModel(nn.Module):
    """MAE Wrapper: Encoder + Masking + Decoder."""
    def __init__(self, encoder: PatchViT, decoder_dim: int = 512, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # Infer patch count
        img_size = encoder.img_size
        patch_size = encoder.patch
        num_patches = (img_size // patch_size) ** 2
        
        self.decoder = MaeDecoder(
            embed_dim=encoder.dim,
            patch_size=patch_size,
            num_patches=num_patches,
            decoder_dim=decoder_dim
        )
        
        # Initialize decoder pos embed (simple sin-cos)
        self.decoder.decoder_pos_embed.data.copy_(
            self._get_2d_sincos_pos_embed(self.decoder.decoder_pos_embed.shape[-1], int(num_patches**0.5), cls_token=True)
        )

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        # Simplified numpy-based init
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder.patch
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        # Mean over patch pixels?
        # We compute MSE per patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def random_masking(self, x, mask_ratio):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # 1. Patch Embed
        x = self.encoder.patch_embed(imgs)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. Add Pos Embed (before masking!)
        # encoder.pos_embed has CLS. We need to handle it.
        # x is (B, N, D). pos_embed is (1, N+1, D).
        pos_embed_sans_cls = self.encoder.pos_embed[:, 1:, :]
        x = x + pos_embed_sans_cls

        # 3. Masking
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # 4. Append CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        # 5. Encoder Blocks
        for blk in self.encoder.blocks:
            x_masked = blk(x_masked)
        x_masked = self.encoder.norm(x_masked)

        # 6. Decoder
        pred = self.decoder(x_masked, ids_restore)
        
        return pred, mask


class _StopFlag:
    """Signal handler flag."""
    def __init__(self):
        self.stop = False


def _seed_all(seed: int) -> None:
    """Seed all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict[str, Any]:
    """Get all RNG states."""
    torch_state = torch.get_rng_state()
    # Convert ByteTensor to CPU for serialization
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.cpu()
    
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch_state,
    }
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        state["cuda"] = [s.cpu() for s in cuda_states]
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    
    # Handle torch RNG state - ensure it's on CPU
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        if torch_state.is_cuda:
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]
        if isinstance(cuda_states, list):
            # Ensure all states are on CPU before setting
            cpu_states = [s.cpu() if isinstance(s, torch.Tensor) and s.is_cuda else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cpu_states)
        else:
            torch.cuda.set_rng_state_all(cuda_states)


def _set_sdp_backend(backend: str, *, is_rocm: bool = False) -> None:
    """Set scaled dot product attention backend.

    On ROCm, "auto" can select flash kernels that are sometimes unstable depending on
    driver/runtime; prefer mem_efficient with math enabled as a fallback.
    """
    if not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        return

    if backend == "auto":
        if is_rocm:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        return

    torch.backends.cuda.enable_flash_sdp(backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(backend == "mem_efficient")

    # On ROCm, keep math enabled as a safe fallback when mem_efficient can't run.
    torch.backends.cuda.enable_math_sdp(backend == "math" or (is_rocm and backend == "mem_efficient"))


def save_checkpoint(
    path: Path,
    step: int,
    student: nn.Module,
    teacher: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dino_loss: DINOLoss,
    config: TrainingConfig,
) -> None:
    """Save training checkpoint."""
    payload = {
        "step": step,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "dino_loss": dino_loss.state_dict(),
        "rng": _get_rng_state(),
        "config": asdict(config),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    student: nn.Module,
    teacher: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dino_loss: DINOLoss,
    device: torch.device,
    scale_aware: bool = False,
) -> tuple[int, TrainingConfig]:
    """Load training checkpoint.

    When loading a non-scale-aware checkpoint into a scale-aware model (or vice
    versa), uses ``strict=False`` so that missing/unexpected ``scale_embed.*``
    keys are silently skipped rather than raising an error.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # PyTorch 2.6+ defaults weights_only=True; we store config/RNG (trusted)
    payload = torch.load(path, map_location=device, weights_only=False)

    # Migrate old-format state dict keys (nn.MultiheadAttention → timm-style)
    for key in ("student", "teacher"):
        if key in payload and needs_migration(payload[key]):
            warnings.warn(f"Migrating old-format {key} state dict keys to timm-style")
            payload[key] = migrate_state_dict(payload[key])

    # Detect scale_aware mismatch between checkpoint and current model
    ckpt_config = payload.get("config", {})
    ckpt_scale_aware = ckpt_config.get("scale_aware", False)
    strict = (ckpt_scale_aware == scale_aware)
    if not strict:
        warnings.warn(
            f"Scale-aware mismatch: checkpoint={ckpt_scale_aware}, current={scale_aware}. "
            "Loading with strict=False (scale_embed weights will be freshly initialized)."
        )
    
    student.load_state_dict(payload["student"], strict=strict)
    teacher.load_state_dict(payload["teacher"], strict=strict)
    opt.load_state_dict(payload["opt"])
    
    if payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    
    if payload.get("dino_loss") is not None:
        dino_loss.load_state_dict(payload["dino_loss"])
    
    if payload.get("rng") is not None:
        _set_rng_state(payload["rng"])
    
    step = int(payload.get("step", 0))
    
    # Reconstruct config
    config_dict = payload.get("config", {})
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    hw_cfg_dict = config_dict.get("hardware")
    hw_cfg = HardwareConfig(**hw_cfg_dict) if hw_cfg_dict else None
    
    config = TrainingConfig(
        model=model_cfg,
        hardware=hw_cfg,
        **{k: v for k, v in config_dict.items() if k not in ["model", "hardware"]}
    )
    
    return step, config


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find latest checkpoint in run directory."""
    ckpts = sorted(run_dir.glob("checkpoint_*.pth"))
    return ckpts[-1] if ckpts else None


def rotate_checkpoints(run_dir: Path, keep_last: int) -> None:
    """Keep only the last N checkpoints."""
    ckpts = sorted(run_dir.glob("checkpoint_*.pth"))
    if len(ckpts) > keep_last:
        for ckpt in ckpts[:-keep_last]:
            ckpt.unlink()


def detect_anomaly(
    loss: float,
    loss_history: list[float],
    embedding_std: float | None = None,
) -> tuple[bool, str | None]:
    """Detect training anomalies."""
    # NaN/Inf check
    if not np.isfinite(loss):
        return True, f"Loss is {'NaN' if np.isnan(loss) else 'Inf'}"
    
    # Loss spike check
    if len(loss_history) >= 10:
        recent_mean = np.mean(loss_history[-10:])
        if loss > recent_mean * 2.0:
            return True, f"Loss spike detected: {loss:.4f} > 2x recent mean {recent_mean:.4f}"
    
    # Feature collapse check
    if embedding_std is not None and embedding_std < 0.01:
        return True, f"Feature collapse detected: embedding std={embedding_std:.6f}"
    
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5: Big Run - Production DINOv3 training")
    
    # Model configuration
    ap.add_argument(
        "--config",
        choices=list(MODEL_CONFIGS.keys()) + ["custom"],
        default="vit-large",
        help="Model configuration preset",
    )
    ap.add_argument("--vit-patch", type=int, help="Custom: patch size")
    ap.add_argument("--vit-dim", type=int, help="Custom: model dimension")
    ap.add_argument("--vit-depth", type=int, help="Custom: number of transformer blocks")
    ap.add_argument("--vit-heads", type=int, help="Custom: number of attention heads")
    ap.add_argument("--out-dim", type=int, help="Override output dimension (default: 8192)")
    
    # Hardware
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Hardware target (auto-detect or override)",
    )
    ap.add_argument("--num-workers", type=int, help="Override num_workers")
    ap.add_argument("--pin-memory", type=bool, help="Override pin_memory")
    
    # Training
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accumulation-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=2500)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--max-steps", type=int, help="Maximum training steps (None = unlimited)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves memory)")
    
    # DINO
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--center-momentum", type=float, default=0.9, help="Momentum for DINO centering (default: 0.9, recommended: 0.999 for stability)")
    
    # Gram anchoring (REQUIRED for medical imaging - DO NOT DISABLE)
    ap.add_argument("--gram-weight", type=float, default=1.0, help="Gram Anchoring weight (default: 1.0)")
    ap.add_argument("--koleo-weight", type=float, default=0.0, help="KoLeo regularization weight (default: 0.0)")
    
    # Loss Type
    ap.add_argument("--loss-type", choices=["dino", "simclr", "mae"], default="dino", help="Objective function")

    # Scale Awareness
    ap.add_argument("--scale-aware", action="store_true",
                    help="Enable scale embedding: injects (pixel_spacing_x, pixel_spacing_y, slice_thickness) "
                         "into the ViT. Requires spacing_x/spacing_y/spacing_z columns in index CSV.")

    # Checkpointing
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--ckpt-keep-last", type=int, default=5)
    ap.add_argument("--resume", type=str, help="Resume from checkpoint ('auto' or path)")
    
    # Monitoring
    ap.add_argument("--monitor-every", type=int, default=1000)
    
    # Data
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--split-manifest", type=Path, help="Split manifest JSON (excludes val set)")
    ap.add_argument("--crop-scale-min", type=float, default=0.3,
                    help="Min scale for RandomResizedCrop (default: 0.3)")
    ap.add_argument("--crop-scale-max", type=float, default=1.0,
                    help="Max scale for RandomResizedCrop (default: 1.0)")
    ap.add_argument("--z-stride", type=int, default=1,
                    help="Keep every Nth slice per series to reduce z-axis correlation "
                         "(default: 1 = all slices, 3 = removes adjacent channel overlap)")
    ap.add_argument("--diverse-batches", action="store_true",
                    help="Use series-diverse batch sampling (at most one sample per series per batch)")
    
    # Reproducibility
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--sdp-backend", choices=["auto", "math", "mem_efficient", "flash"], default="auto")
    
    # Output
    ap.add_argument("--run-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--run-suffix", type=str, help="Optional suffix for run directory name for easier identification (e.g., 'amd395_128x2')")
    
    # AMP
    ap.add_argument("--amp", action="store_true", help="Use mixed precision training")
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16",
                    help="AMP dtype (default: bfloat16 — same exponent range as fp32, "
                         "avoids overflow/underflow in contrastive losses)")

    # JSON-lines metrics log (one line per step, for programmatic consumers)
    ap.add_argument("--log-json", type=Path, default=None,
                    help="Write one JSON line per training step to this file")

    args = ap.parse_args()

    # Resolve AMP dtype
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.amp_dtype]
    # bf16 has the same exponent range as fp32, so GradScaler is unnecessary
    # and can actually hurt training stability. Only enable for fp16.
    use_grad_scaler = args.amp and amp_dtype == torch.float16
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Configuration Setup
    # ─────────────────────────────────────────────────────────────────────────
    
    # Model configuration
    if args.config == "custom":
        if not all([args.vit_patch, args.vit_dim, args.vit_depth, args.vit_heads]):
            raise ValueError("Custom config requires: --vit-patch, --vit-dim, --vit-depth, --vit-heads")
        model_cfg = ModelConfig(
            name="custom",
            patch=args.vit_patch,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
        )
    else:
        model_cfg = MODEL_CONFIGS[args.config]
        # Allow overrides
        overridden = False
        if args.vit_patch:
            model_cfg.patch = args.vit_patch
            overridden = True
        if args.vit_dim:
            model_cfg.dim = args.vit_dim
            overridden = True
        if args.vit_depth:
            model_cfg.depth = args.vit_depth
            overridden = True
        if args.vit_heads:
            model_cfg.heads = args.vit_heads
            overridden = True
        if overridden:
            # Update name to match known preset or mark as custom
            matched = False
            for preset_name, preset_cfg in MODEL_CONFIGS.items():
                if (model_cfg.patch == preset_cfg.patch and
                    model_cfg.dim == preset_cfg.dim and
                    model_cfg.depth == preset_cfg.depth and
                    model_cfg.heads == preset_cfg.heads):
                    model_cfg.name = preset_name
                    matched = True
                    break
            if not matched:
                model_cfg.name = "custom"
    
    # Override out_dim if specified
    if args.out_dim:
        model_cfg.out_dim = args.out_dim

    print(f"model_config={model_cfg.name} patch={model_cfg.patch} dim={model_cfg.dim} "
          f"depth={model_cfg.depth} heads={model_cfg.heads} out_dim={model_cfg.out_dim} "
          f"entropy_wall={math.log(model_cfg.out_dim):.4f} "
          f"params={model_cfg.params_millions:.1f}M "
          f"grad_checkpoint={args.grad_checkpoint}")
    
    # Hardware detection
    hw_cfg = detect_hardware()
    if args.device != "auto":
        hw_cfg.device_type = args.device
    if args.num_workers is not None:
        hw_cfg.num_workers = args.num_workers
    if args.pin_memory is not None:
        hw_cfg.pin_memory = args.pin_memory
    
    print(f"hardware={hw_cfg.device_name} device_type={hw_cfg.device_type} "
          f"is_rocm={hw_cfg.is_rocm} num_workers={hw_cfg.num_workers} "
          f"pin_memory={hw_cfg.pin_memory} batch_size_rec={hw_cfg.batch_size_recommendation}")
    
    # Provenance
    git_commit = get_git_commit()
    data_hash = compute_data_manifest_hash(args.index_csv)
    
    if git_commit:
        print(f"git_commit={git_commit}")
    if data_hash:
        print(f"data_manifest_hash={data_hash}")
    
    # Training configuration
    training_cfg = TrainingConfig(
        model=model_cfg,
        hardware=hw_cfg,
        img_size=args.img_size,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        ema=args.ema,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
        loss_type=args.loss_type,
        gram_enabled=True,  # Always enabled - required for medical imaging
        gram_weight=args.gram_weight,
        koleo_weight=args.koleo_weight,
        scale_aware=args.scale_aware,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        z_stride=args.z_stride,
        diverse_batches=args.diverse_batches,
        ckpt_every=args.ckpt_every,
        ckpt_keep_last=args.ckpt_keep_last,
        monitor_every=args.monitor_every,
        train_seed=args.train_seed,
        sdp_backend=args.sdp_backend,
        amp_dtype=args.amp_dtype,
        index_csv=str(args.index_csv),
        split_manifest=str(args.split_manifest) if args.split_manifest else None,
        git_commit=git_commit,
        data_manifest_hash=data_hash,
    )
    
    print(f"effective_batch_size={training_cfg.effective_batch_size} "
          f"(batch={args.batch_size} × accum={args.accumulation_steps})")
    
    _seed_all(args.train_seed)
    _set_sdp_backend(args.sdp_backend, is_rocm=hw_cfg.is_rocm)
    
    device = torch.device(hw_cfg.device_type)
    print(f"device={device.type}")
    print(f"torch_version={torch.__version__}")
    print(f"amp={args.amp} dtype={args.amp_dtype} grad_scaler={use_grad_scaler}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. Run Directory and Resume Logic
    # ─────────────────────────────────────────────────────────────────────────
    
    resume_from: Path | None = None
    run_dir: Path
    
    if args.resume:
        if args.resume == "auto":
            # Find latest run
            if args.run_dir.exists():
                run_dirs = sorted([d for d in args.run_dir.iterdir() if d.is_dir()])
                if run_dirs:
                    run_dir = run_dirs[-1]
                    resume_from = find_latest_checkpoint(run_dir)
                    if not resume_from:
                        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
                else:
                    raise FileNotFoundError(f"No run directories found in {args.run_dir}")
            else:
                raise FileNotFoundError(f"Run directory does not exist: {args.run_dir}")
        else:
            resume_from = Path(args.resume)
            if args.run_suffix:
                # New directory for a distinct experiment resuming from another run
                run_id = time.strftime("%Y%m%d_%H%M%S")
                run_id = f"{run_id}_{args.run_suffix}"
                run_dir = args.run_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Continue in the same run directory as the checkpoint
                run_dir = resume_from.parent
    else:
        # New run
        run_id = time.strftime("%Y%m%d_%H%M%S")
        if args.run_suffix:
            run_id = f"{run_id}_{args.run_suffix}"
        run_dir = args.run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"run_dir={run_dir}")
    
    # Save config
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(training_cfg), indent=2) + "\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. Data Loading
    # ─────────────────────────────────────────────────────────────────────────
    
    all_rows = _load_index_rows(args.index_csv, require_spacing=args.scale_aware)
    print(f"loaded_rows={len(all_rows)} scale_aware={args.scale_aware}")
    
    # Exclude validation set if split manifest provided
    if args.split_manifest and args.split_manifest.exists():
        payload = json.loads(args.split_manifest.read_text())
        val_series = payload.get("val", {}).get("series_dir", [])
        val_set = set(str(s) for s in val_series)
        before = len(all_rows)
        all_rows = [r for r in all_rows if str(r.series_dir) not in val_set]
        print(f"excluded_val_series={len(val_series)} excluded_rows={before - len(all_rows)}")
    
    # Z-stride: subsample slices to reduce z-axis correlation
    if args.z_stride > 1:
        from collections import defaultdict
        series_rows: dict[str, list] = defaultdict(list)
        for r in all_rows:
            series_rows[r.series_dir].append(r)
        strided = []
        for s_dir in sorted(series_rows):
            rows_sorted = sorted(series_rows[s_dir], key=lambda r: r.slice_index)
            strided.extend(rows_sorted[::args.z_stride])
        print(f"z_stride={args.z_stride} rows_before={len(all_rows)} rows_after={len(strided)}")
        all_rows = strided

    def _worker_init_fn(worker_id: int) -> None:
        _seed_all(args.train_seed + worker_id)
    
    gen = torch.Generator()
    gen.manual_seed(args.train_seed)
    
    ds = PngDataset(
        all_rows,
        img_size=args.img_size,
        rw_level_min=training_cfg.rw_level_min,
        rw_level_max=training_cfg.rw_level_max,
        rw_width_min=training_cfg.rw_width_min,
        rw_width_max=training_cfg.rw_width_max,
        scale_aware=args.scale_aware,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
    )
    
    if len(ds) < args.batch_size:
        print(f"⚠️  Dataset size ({len(ds)}) is smaller than batch size ({args.batch_size}).")
        print(f"    Reducing batch size to {len(ds)} to prevent training failure.")
        args.batch_size = len(ds)
        training_cfg.batch_size = args.batch_size

    if args.diverse_batches:
        batch_sampler = DiverseBatchSampler(
            all_rows, batch_size=args.batch_size, drop_last=True, generator=gen,
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=hw_cfg.num_workers,
            pin_memory=hw_cfg.pin_memory,
            worker_init_fn=_worker_init_fn,
            collate_fn=dino_collate,
        )
        print(f"diverse_batches=True batches_per_epoch={len(batch_sampler)}")
    else:
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=hw_cfg.num_workers,
            pin_memory=hw_cfg.pin_memory,
            drop_last=True,
            generator=gen,
            worker_init_fn=_worker_init_fn,
            collate_fn=dino_collate,
        )
    it = iter(dl)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. Model Setup
    # ─────────────────────────────────────────────────────────────────────────
    
    vit_s = PatchViT(
        img_size=args.img_size,
        patch=model_cfg.patch,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_ratio=model_cfg.mlp_ratio,
        use_grad_checkpoint=args.grad_checkpoint,
        scale_aware=args.scale_aware,
    )
    student = DinoStudentTeacher(vit_s, out_dim=model_cfg.out_dim).to(device)
    
    vit_t = PatchViT(
        img_size=args.img_size,
        patch=model_cfg.patch,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_ratio=model_cfg.mlp_ratio,
        use_grad_checkpoint=args.grad_checkpoint,
        scale_aware=args.scale_aware,
    )
    teacher = DinoStudentTeacher(vit_t, out_dim=model_cfg.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_grad_scaler and device.type == "cuda"))
    dino_loss_fn = DINOLoss(model_cfg.out_dim, center_momentum=training_cfg.center_momentum).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    simclr_loss_fn = SimCLRLoss(temperature=0.1).to(device)
    
    mae_model: MaeModel | None = None
    if args.loss_type == "mae":
        mae_model = MaeModel(vit_s, mask_ratio=0.75).to(device)
        # In MAE mode, we optimize the whole MaeModel (encoder+decoder)
        # Note: We reuse 'vit_s' as the encoder backbone inside MaeModel
        opt = torch.optim.AdamW(mae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # DINO/SimCLR mode
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_grad_scaler and device.type == "cuda"))
    
    start_step = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. Resume from Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    
    if resume_from:
        print(f"resume=true checkpoint={resume_from}")
        start_step, loaded_cfg = load_checkpoint(resume_from, student, teacher, opt, scaler, dino_loss_fn, device, scale_aware=args.scale_aware)
        print(f"resumed_from_step={start_step}")
        
        # Check for hardware change
        if loaded_cfg.hardware and loaded_cfg.hardware.device_name != hw_cfg.device_name:
            print(f"⚠️  Hardware platform changed: {loaded_cfg.hardware.device_name} → {hw_cfg.device_name}")
            print(f"    Applying new optimization presets for {hw_cfg.device_name}")
        
        # Check for config mismatch
        if loaded_cfg.model.name != model_cfg.name:
            warnings.warn(f"Model config mismatch: checkpoint={loaded_cfg.model.name} requested={model_cfg.name}")
    
    # TensorBoard Logger
    tb_writer = SummaryWriter(log_dir=str(run_dir)) if SummaryWriter is not None else None
    if tb_writer:
        print(f"tensorboard_log_dir={run_dir}")
    else:
        print("tensorboard not installed, skipping TB logging")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Training Loop
    # ─────────────────────────────────────────────────────────────────────────
    
    stop = _StopFlag()
    
    def _handle_sigint(_sig: int, _frame: Any) -> None:
        stop.stop = True
    
    signal.signal(signal.SIGINT, _handle_sigint)
    
    loss_history: list[float] = []
    t0 = time.time()
    last_log = t0
    
    max_steps = args.max_steps if args.max_steps else 10**9  # Effectively unlimited
    
    print(f"Starting training from step {start_step} to {max_steps if args.max_steps else 'unlimited'}")
    print("─" * 80)
    
    for step in range(start_step, int(max_steps)):
        if stop.stop:
            print("interrupt=true")
            break
        
        # Update learning rate
        current_lr = get_lr(
            step=step,
            total_steps=training_cfg.max_steps,
            warmup_steps=training_cfg.warmup_steps,
            base_lr=training_cfg.lr,
            min_lr=training_cfg.min_lr,
        )
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr
        
        # Data loading: ([view1, view2], spacing)
        try:
            views, spacing = next(it)
        except StopIteration:
            it = iter(dl)
            views, spacing = next(it)
        
        # views is a list of [batch_v1, batch_v2]
        # Concatenate for efficient forward pass: (2*B, 3, H, W)
        batch = torch.cat(views, dim=0).to(device, non_blocking=True)
        # Duplicate spacing for both views (both come from the same slice)
        spacing_2b = torch.cat([spacing, spacing], dim=0).to(device, non_blocking=True) if args.scale_aware else None
        
        # Forward pass
        amp_enabled = bool(args.amp and device.type == "cuda")
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            loss_dino = torch.tensor(0.0, device=device)
            loss_simclr = torch.tensor(0.0, device=device)
            loss_mae = torch.tensor(0.0, device=device)
            loss_gram = torch.tensor(0.0, device=device)
            loss_koleo = torch.tensor(0.0, device=device)

            if training_cfg.loss_type == "mae":
                pred, mask = mae_model(batch)
                loss = mae_model.forward_loss(batch, pred, mask)
                loss_mae = loss
            elif training_cfg.loss_type == "simclr":
                student_feats = student.backbone(batch, spacing=spacing_2b)
                student_out = student.head(student_feats[:, 0])
                
                # SimCLR takes z1 and z2 from student
                # student_out is (2B, D) -> split into (B, D), (B, D)
                B_sim = student_out.shape[0] // 2
                z1, z2 = student_out[:B_sim], student_out[B_sim:]
                loss = simclr_loss_fn(z1, z2)
                loss_simclr = loss
            else:
                # DINO default
                # Get features for both DINO and Gram losses
                student_feats = student.backbone(batch, spacing=spacing_2b)
                with torch.no_grad():
                    teacher_feats = teacher.backbone(batch, spacing=spacing_2b)

                # DINO loss: use CLS token through projection head
                student_out = student.head(student_feats[:, 0])
                teacher_out = teacher.head(teacher_feats[:, 0])
            
                loss = dino_loss_fn(
                    student_out,
                    teacher_out,
                    args.student_temp,
                    args.teacher_temp,
                )
                loss_dino = loss
                
                # Gram anchoring (optional but enabled)
                if training_cfg.gram_enabled:
                    # Reuse features from above - no double forward
                    loss_gram = compute_gram_anchoring_loss(student_feats, teacher_feats)
                    loss = loss + training_cfg.gram_weight * loss_gram
                
                # KoLeo Regularization
                if training_cfg.koleo_weight > 0.0:
                    loss_koleo = koleo_loss_fn(student_out)
                    loss = loss + training_cfg.koleo_weight * loss_koleo

            # Gradient accumulation
            loss = loss / args.accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # ROCm safety: synchronize after backward to catch errors early
        if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip:
            torch.cuda.synchronize()
        
        # Optimizer step (every accumulation_steps)
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale to check gradients
            scaler.unscale_(opt)
            
            # Compute Grad Norm
            total_norm = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip grads (optional but good for stability)
            # torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            # EMA update teacher
            if args.loss_type == "dino":
                with torch.no_grad():
                    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                        p_t.data.mul_(args.ema).add_(p_s.data, alpha=1.0 - args.ema)
        else:
            total_norm = 0.0 # Placeholder
        
        # Logging
        loss_val = loss.item() * args.accumulation_steps
        loss_history.append(loss_val)

        # Per-step JSON-lines log (lightweight, for programmatic consumers)
        if args.log_json is not None:
            _jl = json.dumps({
                "step": step,
                "loss": round(loss_val, 6),
                "lr": opt.param_groups[0]["lr"],
            })
            with open(args.log_json, "a") as _jf:
                _jf.write(_jl + "\n")
        
        if time.time() - last_log >= 10.0 or step == start_step:
            elapsed = time.time() - t0
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
            samples_per_sec = steps_per_sec * training_cfg.effective_batch_size
            current_lr = opt.param_groups[0]["lr"]
            
            print(
                f"step={step:6d} loss={loss_val:.4f} lr={current_lr:.2e} "
                f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.1f} "
                f"elapsed={elapsed:.1f}s"
            )
            
            # TensorBoard Scalars
            if tb_writer:
                tb_writer.add_scalar("Train/Loss_Total", loss_val, step)
                tb_writer.add_scalar("Train/Loss_DINO", float(loss_dino.item()), step)
                tb_writer.add_scalar("Train/Loss_SimCLR", float(loss_simclr.item()), step)
                tb_writer.add_scalar("Train/Loss_MAE", float(loss_mae.item()), step)
                tb_writer.add_scalar("Train/Loss_Gram", float(loss_gram.item()), step)
                tb_writer.add_scalar("Train/Loss_KoLeo", float(loss_koleo.item()), step)
                if total_norm > 0:
                    tb_writer.add_scalar("Train/Grad_Norm", float(total_norm), step)

            if args.loss_type == "dino":
                with torch.no_grad():
                    # Use F.log_softmax (fused LogSumExp) for numerical stability.
                    # Avoids 0 * -inf = NaN that occurs with separate softmax + log.
                    t_logits = (teacher_out.float() - dino_loss_fn.center.float()) / args.teacher_temp
                    s_logits = student_out.detach().float() / args.student_temp
                    t_ent = -(F.softmax(t_logits, dim=-1) * F.log_softmax(t_logits, dim=-1)).sum(dim=-1).mean().item()
                    s_ent = -(F.softmax(s_logits, dim=-1) * F.log_softmax(s_logits, dim=-1)).sum(dim=-1).mean().item()
                if tb_writer:
                    tb_writer.add_scalar("Train/Entropy_Teacher", t_ent, step)
                    tb_writer.add_scalar("Train/Entropy_Student", s_ent, step)

            if tb_writer:
                tb_writer.add_scalar("Train/LR", current_lr, step)
                tb_writer.add_scalar("Perf/Samples_Per_Sec", samples_per_sec, step)
                tb_writer.flush()

            last_log = time.time()
        
        # Anomaly detection
        is_anomaly, anomaly_msg = detect_anomaly(loss_val, loss_history)
        if is_anomaly:
            if "NaN" in anomaly_msg or "Inf" in anomaly_msg:
                print(f"❌ CRITICAL: {anomaly_msg}")
                print("Saving emergency checkpoint...")
                emergency_path = run_dir / f"emergency_checkpoint_step{step}.pth"
                save_checkpoint(emergency_path, step, student, teacher, opt, scaler, dino_loss_fn, training_cfg)
                raise RuntimeError(anomaly_msg)
            else:
                print(f"⚠️  WARNING: {anomaly_msg}")
        
        # Checkpointing
        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = run_dir / f"checkpoint_{step+1:08d}.pth"
            
            # Save correct model based on mode
            if args.loss_type == "mae":
                _student = mae_model
                _teacher = mae_model # Dummy
            else:
                _student = student
                _teacher = teacher

            save_checkpoint(ckpt_path, step + 1, _student, _teacher, opt, scaler, dino_loss_fn, training_cfg)
            print(f"checkpoint_saved={ckpt_path}")
            
            # Rotate old checkpoints
            rotate_checkpoints(run_dir, args.ckpt_keep_last)
        
        # Monitoring
        if args.monitor_every and (step + 1) % args.monitor_every == 0:
            if args.loss_type == "mae":
                input_slice = batch[0, 1, :, :].detach().cpu().numpy()
                if tb_writer:
                    tb_writer.add_image("Monitor/Input", input_slice, step, dataformats="HW")
            else:
                heatmap = make_attention_heatmap(student, batch)
                if tb_writer:
                    tb_writer.add_image("Monitor/Attention", heatmap, step, dataformats="HW")
            
                # Log input slice (first channel of first image in batch)
                input_slice = batch[0, 1, :, :].detach().cpu().numpy() # Middle slice
                if tb_writer:
                    tb_writer.add_image("Monitor/Input", input_slice, step, dataformats="HW")

                # Create Combined Image (Side-by-Side)
                try:
                    # Convert input slice (float 0-1) to uint8 0-255
                    img_h, img_w = input_slice.shape
                    input_uint8 = (np.clip(input_slice, 0, 1) * 255).astype(np.uint8)
                    input_pil = Image.fromarray(input_uint8, mode='L').convert('RGB')

                    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                    heatmap_pil = Image.fromarray(heatmap_uint8, mode='L').resize((img_w, img_h), resample=Image.Resampling.NEAREST).convert('RGB')

                    # Stitch
                    combined = Image.new('RGB', (img_w * 2, img_h))
                    combined.paste(input_pil, (0, 0))
                    combined.paste(heatmap_pil, (img_w, 0))

                    combined_np = np.array(combined)
                    if tb_writer:
                        tb_writer.add_image("Monitor/Combined", combined_np, step, dataformats="HWC")

                except Exception as e:
                    print(f"⚠️  Monitoring visualization error: {e}")
                
                # Log Gram Matrix
                try:
                    # Debug stats
                    with torch.no_grad():
                        # Re-compute gram for stats (or we could have returned it)
                        model_s = student
                        model_s.eval()
                        
                        # 1. Check Embedding Layer (Start)
                        # We need to manually call patch_embed + pos_embed to check early diversity
                        x_in = batch[0:1]
                        x_emb = model_s.backbone.patch_embed(x_in)
                        x_emb = x_emb.flatten(2).transpose(1, 2)
                        # Add CLS + Pos (simplifying just to check patch diversity)
                        # x_emb is (1, N, D). Let's check std across N.
                        emb_std = x_emb.std(dim=1).mean().item()
                        
                        # 2. Check Backbone Output (End)
                        # Pass spacing if scale_aware (use first sample's spacing)
                        x_spacing = spacing_2b[0:1] if spacing_2b is not None else None
                        feats = model_s.backbone(x_in, spacing=x_spacing)
                        
                        # L2 norm of patch tokens (skip CLS and skip Registers if present)
                        # feats: (1, N_total, D)
                        # Patches are at [1 : 1+N_patches]
                        num_patches = model_s.backbone.patch_embed.grid_size[0] * model_s.backbone.patch_embed.grid_size[1] if hasattr(model_s.backbone.patch_embed, 'grid_size') else int((model_s.backbone.img_size // model_s.backbone.patch)**2)
                        
                        patches = feats[:, 1 : 1 + num_patches, :] # (1, N_patches, D)
                        
                        patches_norm = F.normalize(patches, p=2, dim=-1)
                        gram = torch.bmm(patches_norm, patches_norm.transpose(1, 2)).squeeze(0)
                        
                        g_min, g_max, g_mean = gram.min().item(), gram.max().item(), gram.mean().item()
                        
                        # Input stats
                        img = batch[0]
                        i_min, i_max, i_mean = img.min().item(), img.max().item(), img.mean().item()
                        
                        print(f" [Monitor] Input: min={i_min:.4f} max={i_max:.4f} mean={i_mean:.4f}")
                        print(f" [Monitor] Embed-L0 Diversity: std={emb_std:.6f} (If 0, PatchEmbed is broken)")
                        print(f" [Monitor] Output Gram: mean={g_mean:.4f} (If 1, Attention collapsed)")

                    gram_img = make_gram_heatmap(student, batch)
                    if tb_writer:
                        tb_writer.add_image("Monitor/Gram", gram_img, step, dataformats="HW")
                except Exception as e:
                    print(f"⚠️  Gram visualization error: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. Final Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    
    if tb_writer:
        tb_writer.close()
    
    final_step = step + 1
    final_path = run_dir / f"checkpoint_final_{final_step:08d}.pth"
    
    # Save correct model based on mode
    if args.loss_type == "mae":
        _student = mae_model
        _teacher = mae_model
    else:
        _student = student
        _teacher = teacher

    save_checkpoint(final_path, final_step, _student, _teacher, opt, scaler, dino_loss_fn, training_cfg)
    print(f"final_checkpoint={final_path}")
    
    elapsed = time.time() - t0
    print("─" * 80)
    print(f"Training complete: {final_step - start_step} steps in {elapsed:.1f}s")
    print(f"Final loss: {loss_history[-1] if loss_history else 'N/A':.4f}")


if __name__ == "__main__":
    main()

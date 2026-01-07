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
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    _need("torch")


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
    
    # L2 norm of patch tokens (skip CLS)
    patch_tokens = feats[:, 1:, :] # (1, N, D)
    norms = torch.norm(patch_tokens, dim=-1).squeeze(0) # (N,)
    
    # Normalize to [0, 1]
    norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    
    # Reshape to grid
    grid_size = int(np.sqrt(norms.shape[0]))
    heatmap = norms.reshape(grid_size, grid_size).cpu().numpy()
    
    # Resize to match input image size for better visualization
    # We return the small heatmap; TensorBoard handles resizing if needed,
    # or we can rely on the viewer. 
    return heatmap



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
    
    # Gram anchoring - ALWAYS ENABLED (required for medical imaging)
    # Without Gram Anchoring, the model will collapse on CT scans
    gram_enabled: bool = True  # DO NOT CHANGE - hardcoded to True
    gram_weight: float = 1.0
    
    # Checkpointing
    ckpt_every: int = 100
    ckpt_keep_last: int = 5
    
    # Monitoring
    monitor_every: int = 1000
    
    # Seeds and reproducibility
    train_seed: int = 0
    sdp_backend: str = "auto"
    
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
# Model Architecture (from Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

class PatchViT(nn.Module):
    """Patch-based Vision Transformer."""
    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        assert img_size % patch == 0
        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.use_grad_checkpoint = use_grad_checkpoint

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        n_patches = (img_size // patch) * (img_size // patch)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DinoStudentTeacher(nn.Module):
    """DINO student/teacher wrapper with projection head."""
    def __init__(self, backbone: nn.Module, out_dim: int = 8192) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.dim, backbone.dim),
            nn.GELU(),
            nn.Linear(backbone.dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        cls_token = feats[:, 0]
        return self.head(cls_token)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (from Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexRow:
    png_path: Path
    series_dir: str
    slice_index: int
    encoding: str


def _load_index_rows(index_csv: Path) -> list[IndexRow]:
    """Load index CSV."""
    rows = []
    with open(index_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                IndexRow(
                    png_path=Path(r["png_path"]),
                    series_dir=r["series_dir"],
                    slice_index=int(r["slice_index"]),
                    encoding=r["encoding"],
                )
            )
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
    ):
        self.rows = rows
        self.img_size = img_size
        self.rw_level_min = rw_level_min
        self.rw_level_max = rw_level_max
        self.rw_width_min = rw_width_min
        self.rw_width_max = rw_width_max

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
        
        # Resize to model input size
        if windowed.shape != (self.img_size, self.img_size):
            img_pil = Image.fromarray((windowed * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            windowed = np.array(img_pil, dtype=np.float32) / 255.0
            
        return windowed

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
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
            return torch.from_numpy(x).contiguous()

        # Return two different windowed views for DINO cross-view prediction
        return [_get_view(), _get_view()]


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
    # or if we are past total_steps
    if total_steps is None or step >= total_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


class DINOLoss(nn.Module):
    """DINO loss with centering and sharpening to prevent collapse."""
    def __init__(self, out_dim: int, center_momentum: float = 0.9) -> None:
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


def _set_sdp_backend(backend: str) -> None:
    """Set scaled dot product attention backend."""
    if backend == "auto":
        return
    
    if not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        return
    
    torch.backends.cuda.enable_flash_sdp(backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(backend == "mem_efficient")
    torch.backends.cuda.enable_math_sdp(backend == "math")


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
) -> tuple[int, TrainingConfig]:
    """Load training checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # PyTorch 2.6+ defaults weights_only=True; we store config/RNG (trusted)
    payload = torch.load(path, map_location=device, weights_only=False)
    
    student.load_state_dict(payload["student"], strict=True)
    teacher.load_state_dict(payload["teacher"], strict=True)
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
    
    # Gram anchoring (REQUIRED for medical imaging - DO NOT DISABLE)
    ap.add_argument("--gram-weight", type=float, default=1.0, help="Gram Anchoring weight (default: 1.0)")
    
    # Checkpointing
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--ckpt-keep-last", type=int, default=5)
    ap.add_argument("--resume", type=str, help="Resume from checkpoint ('auto' or path)")
    
    # Monitoring
    ap.add_argument("--monitor-every", type=int, default=1000)
    
    # Data
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--split-manifest", type=Path, help="Split manifest JSON (excludes val set)")
    
    # Reproducibility
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--sdp-backend", choices=["auto", "math", "mem_efficient", "flash"], default="auto")
    
    # Output
    ap.add_argument("--run-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--run-suffix", type=str, help="Optional suffix for run directory name for easier identification (e.g., 'amd395_128x2')")
    
    # AMP
    ap.add_argument("--amp", action="store_true", help="Use mixed precision training")
    
    args = ap.parse_args()
    
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
        if args.vit_patch:
            model_cfg.patch = args.vit_patch
        if args.vit_dim:
            model_cfg.dim = args.vit_dim
        if args.vit_depth:
            model_cfg.depth = args.vit_depth
        if args.vit_heads:
            model_cfg.heads = args.vit_heads
    
    print(f"model_config={model_cfg.name} patch={model_cfg.patch} dim={model_cfg.dim} "
          f"depth={model_cfg.depth} heads={model_cfg.heads} params={model_cfg.params_millions:.1f}M "
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
        gram_enabled=True,  # Always enabled - required for medical imaging
        gram_weight=args.gram_weight,
        ckpt_every=args.ckpt_every,
        ckpt_keep_last=args.ckpt_keep_last,
        monitor_every=args.monitor_every,
        train_seed=args.train_seed,
        sdp_backend=args.sdp_backend,
        index_csv=str(args.index_csv),
        split_manifest=str(args.split_manifest) if args.split_manifest else None,
        git_commit=git_commit,
        data_manifest_hash=data_hash,
    )
    
    print(f"effective_batch_size={training_cfg.effective_batch_size} "
          f"(batch={args.batch_size} × accum={args.accumulation_steps})")
    
    _seed_all(args.train_seed)
    _set_sdp_backend(args.sdp_backend)
    
    device = torch.device(hw_cfg.device_type)
    print(f"device={device.type}")
    print(f"torch_version={torch.__version__}")
    
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
    
    all_rows = _load_index_rows(args.index_csv)
    print(f"loaded_rows={len(all_rows)}")
    
    # Exclude validation set if split manifest provided
    if args.split_manifest and args.split_manifest.exists():
        payload = json.loads(args.split_manifest.read_text())
        val_series = payload.get("val", {}).get("series_dir", [])
        val_set = set(str(s) for s in val_series)
        before = len(all_rows)
        all_rows = [r for r in all_rows if str(r.series_dir) not in val_set]
        print(f"excluded_val_series={len(val_series)} excluded_rows={before - len(all_rows)}")
    
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
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=hw_cfg.num_workers,
        pin_memory=hw_cfg.pin_memory,
        drop_last=True,
        generator=gen,
        worker_init_fn=_worker_init_fn,
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
    )
    teacher = DinoStudentTeacher(vit_t, out_dim=model_cfg.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    dino_loss_fn = DINOLoss(model_cfg.out_dim).to(device)
    
    start_step = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. Resume from Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    
    if resume_from:
        print(f"resume=true checkpoint={resume_from}")
        start_step, loaded_cfg = load_checkpoint(resume_from, student, teacher, opt, scaler, dino_loss_fn, device)
        print(f"resumed_from_step={start_step}")
        
        # Check for hardware change
        if loaded_cfg.hardware and loaded_cfg.hardware.device_name != hw_cfg.device_name:
            print(f"⚠️  Hardware platform changed: {loaded_cfg.hardware.device_name} → {hw_cfg.device_name}")
            print(f"    Applying new optimization presets for {hw_cfg.device_name}")
        
        # Check for config mismatch
        if loaded_cfg.model.name != model_cfg.name:
            warnings.warn(f"Model config mismatch: checkpoint={loaded_cfg.model.name} requested={model_cfg.name}")
    
    # TensorBoard Logger
    tb_writer = SummaryWriter(log_dir=str(run_dir))
    print(f"tensorboard_log_dir={run_dir}")

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
        
        # Data loading: list[tensor, tensor]
        try:
            views = next(it)
        except StopIteration:
            it = iter(dl)
            views = next(it)
        
        # views is a list of [batch_v1, batch_v2]
        # Concatenate for efficient forward pass: (2*B, 3, H, W)
        batch = torch.cat(views, dim=0).to(device, non_blocking=True)
        
        # Forward pass
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            student_out = student(batch)
            with torch.no_grad():
                teacher_out = teacher(batch)
            
            # DINO loss with centering, sharpening, and cross-view prediction
            loss = dino_loss_fn(
                student_out,
                teacher_out,
                args.student_temp,
                args.teacher_temp,
            )
            
            # Gram anchoring (optional but enabled)
            loss_gram = torch.tensor(0.0, device=device)
            if training_cfg.gram_enabled:
                student_feats = student.backbone(batch)
                with torch.no_grad():
                    teacher_feats = teacher.backbone(batch)
                # Compute Gram loss on all views in the batch
                loss_gram = compute_gram_anchoring_loss(student_feats, teacher_feats)
                loss = loss + training_cfg.gram_weight * loss_gram
            
            # Gradient accumulation
            loss = loss / args.accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Optimizer step (every accumulation_steps)
        if (step + 1) % args.accumulation_steps == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            # EMA update teacher
            with torch.no_grad():
                for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(args.ema).add_(p_s.data, alpha=1.0 - args.ema)
        
        # Logging
        loss_val = loss.item() * args.accumulation_steps
        loss_history.append(loss_val)
        
        if time.time() - last_log >= 10.0 or step == start_step:
            elapsed = time.time() - t0
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
            samples_per_sec = steps_per_sec * training_cfg.effective_batch_size
            current_lr = opt.param_groups[0]["lr"]
            
            print(
                f"step={step:6d} loss={loss_val:.4f} "
                f"steps/s={steps_per_sec:.2f} samples/s={samples_per_sec:.1f} "
                f"elapsed={elapsed:.1f}s"
            )
            
            # TensorBoard Scalars
            tb_writer.add_scalar("Train/Loss_Total", loss_val, step)
            tb_writer.add_scalar("Train/Loss_DINO", loss_val - training_cfg.gram_weight * loss_gram.item(), step)
            tb_writer.add_scalar("Train/Loss_Gram", loss_gram.item(), step)

            with torch.no_grad():
                t_prob = F.softmax((teacher_out - dino_loss_fn.center) / args.teacher_temp, dim=-1)
                s_prob = F.softmax(student_out.detach() / args.student_temp, dim=-1)
                t_ent = (-t_prob * torch.log(t_prob.clamp_min(1e-12))).sum(dim=-1).mean().item()
                s_ent = (-s_prob * torch.log(s_prob.clamp_min(1e-12))).sum(dim=-1).mean().item()
            tb_writer.add_scalar("Train/Entropy_Teacher", t_ent, step)
            tb_writer.add_scalar("Train/Entropy_Student", s_ent, step)

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
            save_checkpoint(ckpt_path, step + 1, student, teacher, opt, scaler, dino_loss_fn, training_cfg)
            print(f"checkpoint_saved={ckpt_path}")
            
            # Rotate old checkpoints
            rotate_checkpoints(run_dir, args.ckpt_keep_last)
        
        # Monitoring
        if args.monitor_every and (step + 1) % args.monitor_every == 0:
            heatmap = make_attention_heatmap(student, batch)
            tb_writer.add_image("Monitor/Attention", heatmap, step, dataformats="HW")
            
            # Log input slice (first channel of first image in batch)
            input_slice = batch[0, 1, :, :].detach().cpu().numpy() # Middle slice
            tb_writer.add_image("Monitor/Input", input_slice, step, dataformats="HW")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. Final Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    
    tb_writer.close()
    
    final_step = step + 1
    final_path = run_dir / f"checkpoint_final_{final_step:08d}.pth"
    save_checkpoint(final_path, final_step, student, teacher, opt, scaler, dino_loss_fn, training_cfg)
    print(f"final_checkpoint={final_path}")
    
    elapsed = time.time() - t0
    print("─" * 80)
    print(f"Training complete: {final_step - start_step} steps in {elapsed:.1f}s")
    print(f"Final loss: {loss_history[-1] if loss_history else 'N/A':.4f}")


if __name__ == "__main__":
    main()

"""DINO-X model architecture: PatchViT with ScaleEmbedding.

This module is the canonical source for all DINO-X model definitions.
Training scripts, evaluation tools, and the hub API all import from here.

Architecture uses timm-style naming conventions (qkv, proj, fc1, fc2)
for compatibility with HuggingFace peft / LoRA injection.

Attention uses ``F.scaled_dot_product_attention`` to automatically route
to FlashAttention / memory-efficient kernels when available.
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention (timm-style: explicit nn.Linear for peft targeting)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention with explicit QKV projection.

    Uses ``F.scaled_dot_product_attention`` for FlashAttention / mem-efficient
    kernel dispatch. Linear layers are named ``qkv`` and ``proj`` for direct
    peft ``target_modules`` matching.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# MLP (timm-style: named fc1/fc2 for peft targeting)
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation.

    Layers named ``fc1`` / ``fc2`` for peft ``target_modules`` matching.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with timm-style sub-modules."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Scale Embedding
# ---------------------------------------------------------------------------


class ScaleEmbedding(nn.Module):
    """Projects physical spacing (pixel_spacing_x, pixel_spacing_y, slice_thickness) → embed_dim.

    Enables scale-aware representation learning: the model natively knows that a
    14×14 patch at 0.5 mm covers 7 mm of tissue, while the same patch at 1.5 mm
    covers 21 mm. Spacing is treated as a continuous signal (not categorical) so
    the model can generalize to unseen resolutions.

    The output Linear is **zero-initialized** so that:
    - A freshly created ScaleEmbedding produces all-zero vectors.
    - Checkpoints trained *without* scale awareness resume identically when the
      module is added (gradual learning, no sudden perturbation).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = max(embed_dim // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Zero-init the output projection so the embedding starts as a no-op.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, spacing: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spacing: (B, 3) — [pixel_spacing_x, pixel_spacing_y, slice_thickness] in mm.

        Returns:
            (B, 1, embed_dim) — broadcast-ready for addition to patch embeddings.
        """
        return self.mlp(spacing).unsqueeze(1)


# ---------------------------------------------------------------------------
# PatchViT
# ---------------------------------------------------------------------------


class PatchViT(nn.Module):
    """Patch-based Vision Transformer with optional ScaleEmbedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        use_grad_checkpoint: bool = False,
        num_registers: int = 4,
        scale_aware: bool = False,
    ) -> None:
        super().__init__()
        assert img_size % patch == 0
        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_registers = num_registers
        self.scale_aware = scale_aware

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        n_patches = (img_size // patch) * (img_size // patch)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, dim))

        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, dim))

        if scale_aware:
            self.scale_embed = ScaleEmbedding(dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

        # Re-apply zero-init for ScaleEmbedding output projection AFTER
        # self.apply() which would have overwritten it with xavier_uniform.
        if scale_aware:
            nn.init.zeros_(self.scale_embed.mlp[2].weight)
            nn.init.zeros_(self.scale_embed.mlp[2].bias)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        # Specific init for tokens
        nn.init.trunc_normal_(self.pos_embed, std=0.1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

        if self.num_registers > 0:
            nn.init.trunc_normal_(self.registers, std=0.02)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Scale embedding: add physical spacing information to all tokens
        if self.scale_aware and spacing is not None:
            x = x + self.scale_embed(spacing)

        if self.num_registers > 0:
            regs = self.registers.expand(B, -1, -1)
            x = torch.cat([x, regs], dim=1)

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# DINO Student/Teacher wrapper
# ---------------------------------------------------------------------------


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

    def forward(self, x: torch.Tensor, spacing: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone(x, spacing=spacing)
        cls_token = feats[:, 0]
        return self.head(cls_token)


# ---------------------------------------------------------------------------
# Checkpoint state dict migration
# ---------------------------------------------------------------------------

# Patterns for nn.MultiheadAttention → timm-style Attention
_ATTN_KEY_MAP = {
    "in_proj_weight": "qkv.weight",
    "in_proj_bias": "qkv.bias",
    "out_proj.weight": "proj.weight",
    "out_proj.bias": "proj.bias",
}

# Patterns for nn.Sequential MLP → timm-style Mlp
_MLP_KEY_MAP = {
    "0.weight": "fc1.weight",
    "0.bias": "fc1.bias",
    "2.weight": "fc2.weight",
    "2.bias": "fc2.bias",
}

# Regex for attention keys: any prefix + .attn. + old key
_ATTN_RE = re.compile(r"^(.+\.attn)\.(in_proj_weight|in_proj_bias|out_proj\.weight|out_proj\.bias)$")
# Regex for TransformerBlock MLP keys: blocks.N.mlp. + sequential index
# Excludes scale_embed.mlp which uses nn.Sequential (not timm Mlp)
_MLP_RE = re.compile(r"^((?:.*\.)?blocks\.\d+\.mlp)\.(0\.weight|0\.bias|2\.weight|2\.bias)$")


def migrate_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Migrate old-format state dict keys to timm-style naming.

    Handles:
    - ``nn.MultiheadAttention`` keys (``in_proj_weight``, ``out_proj.*``)
      → timm-style ``Attention`` keys (``qkv.*``, ``proj.*``)
    - ``nn.Sequential`` MLP keys (``0.weight``, ``2.weight``)
      → timm-style ``Mlp`` keys (``fc1.weight``, ``fc2.weight``)

    Works with any key prefix (bare ``blocks.N...``, ``backbone.blocks.N...``,
    ``student.backbone.blocks.N...``, etc.).

    Returns a new dict (original is not modified). Keys that don't match
    migration patterns are passed through unchanged.
    """
    new_sd: dict[str, torch.Tensor] = OrderedDict()
    migrated = 0

    for key, value in state_dict.items():
        new_key = key

        # Attention migration
        m = _ATTN_RE.match(key)
        if m:
            prefix, old_suffix = m.group(1), m.group(2)
            new_key = f"{prefix}.{_ATTN_KEY_MAP[old_suffix]}"
            migrated += 1
        else:
            # MLP migration
            m = _MLP_RE.match(key)
            if m:
                prefix, old_suffix = m.group(1), m.group(2)
                new_key = f"{prefix}.{_MLP_KEY_MAP[old_suffix]}"
                migrated += 1

        new_sd[new_key] = value

    return new_sd


def needs_migration(state_dict: dict[str, torch.Tensor]) -> bool:
    """Check if a state dict has old-format keys that need migration."""
    for key in state_dict:
        if _ATTN_RE.match(key) or _MLP_RE.match(key):
            return True
    return False

"""Tests for zoo.arch, zoo.hub, zoo.encode, and zoo.peft.

Covers:
- Architecture forward pass and shapes
- Checkpoint state dict migration (old nn.MultiheadAttention → timm-style)
- Hub: save/load round-trip (training checkpoint + hub format)
- Encode: zero-preprocessing API with various input formats
- PEFT: LoRA injection, gradient isolation, ScaleEmbedding freeze
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zoo.arch import (
    Attention,
    DinoStudentTeacher,
    Mlp,
    PatchViT,
    ScaleEmbedding,
    TransformerBlock,
    migrate_state_dict,
    needs_migration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAttention:
    def test_output_shape(self):
        attn = Attention(dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_named_modules_for_peft(self):
        """qkv and proj must be discoverable by peft target_modules."""
        attn = Attention(dim=64, num_heads=4)
        module_names = {name for name, _ in attn.named_modules() if name}
        assert "qkv" in module_names
        assert "proj" in module_names


class TestMlp:
    def test_output_shape(self):
        mlp = Mlp(dim=64, mlp_ratio=4.0)
        x = torch.randn(2, 10, 64)
        out = mlp(x)
        assert out.shape == (2, 10, 64)

    def test_named_params_for_peft(self):
        """fc1 and fc2 must be discoverable."""
        mlp = Mlp(dim=64)
        param_names = {name.split(".")[0] for name, _ in mlp.named_parameters()}
        assert "fc1" in param_names
        assert "fc2" in param_names


class TestPatchViT:
    def test_forward_no_scale(self):
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        n_patches = (32 // 16) ** 2
        assert out.shape == (2, 1 + n_patches, 64)

    def test_forward_with_scale(self):
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, scale_aware=True, num_registers=0)
        x = torch.randn(2, 3, 32, 32)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 3.0]])
        out = vit(x, spacing=spacing)
        n_patches = (32 // 16) ** 2
        assert out.shape == (2, 1 + n_patches, 64)

    def test_forward_with_registers(self):
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=4)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        n_patches = (32 // 16) ** 2
        assert out.shape == (2, 1 + n_patches + 4, 64)

    def test_all_peft_targets_discoverable(self):
        """All LoRA target modules must exist in the model."""
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4)
        targets = {"qkv", "proj", "fc1", "fc2"}
        found = set()
        for name, _ in vit.named_modules():
            parts = name.split(".")
            found.update(parts)
        assert targets.issubset(found), f"Missing targets: {targets - found}"


# ─────────────────────────────────────────────────────────────────────────────
# State dict migration tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMigration:
    def _make_old_style_sd(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Create a state dict with old nn.MultiheadAttention + Sequential MLP keys."""
        sd = {}
        dim = 64
        mlp_dim = 256
        for i in range(2):
            p = f"{prefix}blocks.{i}" if prefix else f"blocks.{i}"
            # Old attention keys
            sd[f"{p}.attn.in_proj_weight"] = torch.randn(3 * dim, dim)
            sd[f"{p}.attn.in_proj_bias"] = torch.randn(3 * dim)
            sd[f"{p}.attn.out_proj.weight"] = torch.randn(dim, dim)
            sd[f"{p}.attn.out_proj.bias"] = torch.randn(dim)
            # Old MLP keys (Sequential indices)
            sd[f"{p}.mlp.0.weight"] = torch.randn(mlp_dim, dim)
            sd[f"{p}.mlp.0.bias"] = torch.randn(mlp_dim)
            sd[f"{p}.mlp.2.weight"] = torch.randn(dim, mlp_dim)
            sd[f"{p}.mlp.2.bias"] = torch.randn(dim)
            # Norms (unchanged)
            sd[f"{p}.norm1.weight"] = torch.ones(dim)
            sd[f"{p}.norm1.bias"] = torch.zeros(dim)
            sd[f"{p}.norm2.weight"] = torch.ones(dim)
            sd[f"{p}.norm2.bias"] = torch.zeros(dim)
        return sd

    def test_needs_migration_old_keys(self):
        sd = self._make_old_style_sd()
        assert needs_migration(sd)

    def test_needs_migration_new_keys(self):
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        assert not needs_migration(vit.state_dict())

    def test_migrate_bare_keys(self):
        old = self._make_old_style_sd()
        new = migrate_state_dict(old)
        assert "blocks.0.attn.qkv.weight" in new
        assert "blocks.0.attn.proj.weight" in new
        assert "blocks.0.mlp.fc1.weight" in new
        assert "blocks.0.mlp.fc2.weight" in new
        assert "blocks.0.attn.in_proj_weight" not in new
        assert "blocks.0.mlp.0.weight" not in new

    def test_migrate_prefixed_keys(self):
        old = self._make_old_style_sd(prefix="backbone.")
        new = migrate_state_dict(old)
        assert "backbone.blocks.0.attn.qkv.weight" in new
        assert "backbone.blocks.0.attn.proj.weight" in new
        assert "backbone.blocks.0.mlp.fc1.weight" in new

    def test_migrate_preserves_values(self):
        old = self._make_old_style_sd()
        sentinel = torch.tensor([42.0, 43.0])
        old["blocks.0.attn.in_proj_weight"][:2, 0] = sentinel
        new = migrate_state_dict(old)
        assert torch.equal(new["blocks.0.attn.qkv.weight"][:2, 0], sentinel)

    def test_migrate_passthrough_unmatched(self):
        old = self._make_old_style_sd()
        old["cls_token"] = torch.randn(1, 1, 64)
        new = migrate_state_dict(old)
        assert "cls_token" in new
        assert torch.equal(old["cls_token"], new["cls_token"])

    def test_migrated_sd_loads_into_model(self):
        """Old-format state dict should load after migration."""
        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        new_sd = vit.state_dict()

        # Simulate old-format keys from an nn.MultiheadAttention model
        old_sd = {}
        for k, v in new_sd.items():
            old_k = k
            old_k = old_k.replace(".attn.qkv.", ".attn.in_proj_")
            old_k = old_k.replace(".attn.proj.", ".attn.out_proj.")
            old_k = old_k.replace(".mlp.fc1.", ".mlp.0.")
            old_k = old_k.replace(".mlp.fc2.", ".mlp.2.")
            old_sd[old_k] = v

        migrated = migrate_state_dict(old_sd)
        # Should load without errors
        vit.load_state_dict(migrated, strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hub tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHub:
    def test_export_and_load_hub_format(self):
        from zoo.hub import export_hub_checkpoint, load_from_hub_dir

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        x = torch.randn(1, 3, 32, 32)
        expected = vit(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_hub_checkpoint(vit, tmpdir)
            assert (Path(tmpdir) / "config.json").exists()
            assert (Path(tmpdir) / "backbone.pth").exists()

            loaded = load_from_hub_dir(tmpdir)
            actual = loaded(x)
            assert torch.allclose(expected, actual, atol=1e-6)

    def test_load_from_training_checkpoint(self):
        from zoo.hub import load_from_training_checkpoint

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        student = DinoStudentTeacher(vit, out_dim=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            torch.save({
                "student": student.state_dict(),
                "config": {
                    "img_size": 32,
                    "model": {"name": "test", "patch": 16, "dim": 64, "depth": 2, "heads": 4},
                },
                "step": 100,
            }, ckpt_path)

            loaded = load_from_training_checkpoint(ckpt_path)
            assert isinstance(loaded, PatchViT)
            x = torch.randn(1, 3, 32, 32)
            out = loaded(x)
            assert out.shape[2] == 64  # dim

    def test_load_model_unified(self):
        from zoo.hub import export_hub_checkpoint, load_model

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_hub_checkpoint(vit, tmpdir)
            loaded = load_model(tmpdir)
            assert isinstance(loaded, PatchViT)

    def test_load_model_from_pth(self):
        from zoo.hub import load_model

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        student = DinoStudentTeacher(vit, out_dim=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            torch.save({
                "student": student.state_dict(),
                "config": {
                    "img_size": 32,
                    "model": {"name": "test", "patch": 16, "dim": 64, "depth": 2, "heads": 4},
                },
                "step": 100,
            }, ckpt_path)

            loaded = load_model(str(ckpt_path))
            assert isinstance(loaded, PatchViT)


# ─────────────────────────────────────────────────────────────────────────────
# Encode tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEncode:
    def _model(self, scale_aware: bool = False) -> PatchViT:
        return PatchViT(
            img_size=32, patch=16, dim=64, depth=2, heads=4,
            num_registers=0, scale_aware=scale_aware,
        ).eval()

    def test_encode_hu_float_2d(self):
        from zoo.encode import encode
        model = self._model()
        img = np.random.randn(64, 64).astype(np.float32) * 100  # HU values
        feat = encode(model, img, input_format="hu_float")
        assert feat.shape == (1, 1, 64)

    def test_encode_hu16_png(self):
        from zoo.encode import encode
        model = self._model()
        # Simulate 16-bit PNG: HU=0 → uint16=32768
        img = np.full((64, 64), 32768, dtype=np.uint16)
        feat = encode(model, img, input_format="hu16_png")
        assert feat.shape == (1, 1, 64)

    def test_encode_windowed_float(self):
        from zoo.encode import encode
        model = self._model()
        img = np.random.rand(64, 64).astype(np.float32)  # [0, 1]
        feat = encode(model, img, input_format="windowed_float")
        assert feat.shape == (1, 1, 64)

    def test_encode_3channel_hwc(self):
        from zoo.encode import encode
        model = self._model()
        img = np.random.randn(64, 64, 3).astype(np.float32) * 100
        feat = encode(model, img, input_format="hu_float")
        assert feat.shape == (1, 1, 64)

    def test_encode_3channel_chw(self):
        from zoo.encode import encode
        model = self._model()
        img = np.random.randn(3, 64, 64).astype(np.float32) * 100
        feat = encode(model, img, input_format="hu_float")
        assert feat.shape == (1, 1, 64)

    def test_encode_with_spacing(self):
        from zoo.encode import encode
        model = self._model(scale_aware=True)
        img = np.random.randn(64, 64).astype(np.float32) * 100
        feat = encode(model, img, pixel_spacing=(0.5, 0.5), slice_thickness=1.0, input_format="hu_float")
        assert feat.shape == (1, 1, 64)

    def test_encode_spacing_changes_output(self):
        """Different spacings should produce different features for scale-aware models."""
        from zoo.encode import encode
        model = self._model(scale_aware=True)
        # Re-init scale_embed to non-zero so it has an effect
        nn.init.xavier_uniform_(model.scale_embed.mlp[2].weight)

        img = np.random.randn(64, 64).astype(np.float32) * 100
        feat1 = encode(model, img, pixel_spacing=(0.5, 0.5), slice_thickness=1.0, input_format="hu_float")
        feat2 = encode(model, img, pixel_spacing=(2.0, 2.0), slice_thickness=5.0, input_format="hu_float")
        assert not torch.allclose(feat1, feat2, atol=1e-4)

    def test_encode_all_tokens(self):
        from zoo.encode import encode
        model = self._model()
        img = np.random.randn(64, 64).astype(np.float32) * 100
        feat = encode(model, img, return_all_tokens=True, input_format="hu_float")
        n_patches = (32 // 16) ** 2
        assert feat.shape == (1, 1 + n_patches, 64)

    def test_encode_batch(self):
        from zoo.encode import encode_batch
        model = self._model(scale_aware=True)
        images = [np.random.randn(64, 64).astype(np.float32) * 100 for _ in range(3)]
        spacings = [(0.5, 0.5, 1.0), (1.0, 1.0, 2.0), (1.5, 1.5, 3.0)]
        feat = encode_batch(model, images, spacings, input_format="hu_float")
        assert feat.shape == (3, 1, 64)


# ─────────────────────────────────────────────────────────────────────────────
# PEFT tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPeft:
    def _check_peft_available(self):
        try:
            import peft  # noqa: F401
            return True
        except ImportError:
            return False

    def test_apply_lora_gradient_isolation(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        wrapped = apply_lora(vit, rank=4)

        # Only LoRA params should have grad
        trainable_names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        assert len(trainable_names) > 0
        assert all("lora" in n.lower() for n in trainable_names), \
            f"Non-LoRA trainable params: {[n for n in trainable_names if 'lora' not in n.lower()]}"

    def test_apply_lora_scale_embed_frozen(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, scale_aware=True, num_registers=0)
        wrapped = apply_lora(vit, rank=4)

        # ScaleEmbedding must be frozen
        for name, param in wrapped.named_parameters():
            if "scale_embed" in name:
                assert not param.requires_grad, f"ScaleEmbedding param {name} should be frozen"

    def test_apply_lora_patch_embed_frozen(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        wrapped = apply_lora(vit, rank=4)

        for name, param in wrapped.named_parameters():
            if "patch_embed" in name:
                assert not param.requires_grad, f"patch_embed param {name} should be frozen"

    def test_apply_lora_forward_works(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        wrapped = apply_lora(vit, rank=4)

        x = torch.randn(2, 3, 32, 32)
        out = wrapped(x)
        n_patches = (32 // 16) ** 2
        assert out.shape == (2, 1 + n_patches, 64)

    def test_apply_lora_with_scale_aware_forward(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, scale_aware=True, num_registers=0)
        wrapped = apply_lora(vit, rank=4)

        x = torch.randn(2, 3, 32, 32)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 3.0]])
        out = wrapped(x, spacing=spacing)
        n_patches = (32 // 16) ** 2
        assert out.shape == (2, 1 + n_patches, 64)

    def test_save_load_adapter_roundtrip(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora, load_adapter, save_adapter

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        # Save base weights before LoRA wrapping
        base_sd = {k: v.clone() for k, v in vit.state_dict().items()}

        wrapped = apply_lora(vit, rank=4)

        x = torch.randn(1, 3, 32, 32)
        expected = wrapped(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_adapter(wrapped, tmpdir)

            # Reconstruct: fresh backbone with same base weights
            fresh_vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
            fresh_vit.load_state_dict(base_sd)
            reloaded = load_adapter(fresh_vit, tmpdir)
            actual = reloaded(x)
            assert torch.allclose(expected, actual, atol=1e-5)

    def test_count_parameters(self):
        if not self._check_peft_available():
            import pytest
            pytest.skip("peft not installed")

        from zoo.peft import apply_lora, count_parameters

        vit = PatchViT(img_size=32, patch=16, dim=64, depth=2, heads=4, num_registers=0)
        base_params = sum(p.numel() for p in vit.parameters())
        wrapped = apply_lora(vit, rank=4)
        counts = count_parameters(wrapped)
        assert counts["total"] > base_params  # LoRA adds params
        assert counts["trainable"] < counts["total"]  # Most are frozen
        assert counts["trainable"] > 0  # Some are trainable

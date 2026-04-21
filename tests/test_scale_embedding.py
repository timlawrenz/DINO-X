"""Tests for ScaleEmbedding and scale-aware ViT integration.

Validates:
- ScaleEmbedding output shapes and zero-init behavior
- PatchViT backward compatibility (scale_aware=False)
- PatchViT with scale_aware=True (spacing passed / not passed)
- DinoStudentTeacher spacing passthrough
- Gradient flow through the scale embedding
- Collate function correctness
- IndexRow spacing fields and CSV loading
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Import from phase5_big_run — it's a script, not a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from phase5_big_run import (
    DinoStudentTeacher,
    IndexRow,
    PatchViT,
    PngDataset,
    ScaleEmbedding,
    _load_index_rows,
    dino_collate,
)


class TestScaleEmbedding:
    """Test the ScaleEmbedding module in isolation."""

    def test_output_shape(self):
        se = ScaleEmbedding(embed_dim=384)
        spacing = torch.tensor([[0.5, 0.5, 1.0]])  # (1, 3)
        out = se(spacing)
        assert out.shape == (1, 1, 384), f"Expected (1, 1, 384), got {out.shape}"

    def test_batch_output_shape(self):
        se = ScaleEmbedding(embed_dim=384)
        spacing = torch.randn(8, 3)  # (B=8, 3)
        out = se(spacing)
        assert out.shape == (8, 1, 384)

    def test_zero_init_produces_zeros(self):
        """Fresh ScaleEmbedding should produce all-zero outputs (zero-init on output Linear)."""
        se = ScaleEmbedding(embed_dim=384)
        spacing = torch.tensor([[0.5, 0.5, 1.0]])
        out = se(spacing)
        # The output is LayerNorm'd, so it won't be exactly zero
        # But because the Linear before LayerNorm outputs zeros, LayerNorm
        # will produce a constant vector (bias-only). Check it's very small.
        # Actually: zeros through LayerNorm → all zeros → LayerNorm divides by eps → near-zero
        # With zero input, LayerNorm should produce (0 - 0) / eps * weight + bias
        # Default LN: weight=1, bias=0, so output ≈ 0
        assert out.abs().max().item() < 1e-3, f"Zero-init should produce near-zero output, got max={out.abs().max()}"

    def test_different_spacings_produce_different_outputs_after_training(self):
        """After some gradient steps, different spacings should produce different embeddings."""
        se = ScaleEmbedding(embed_dim=64)
        # Initialize with non-zero weights to simulate post-training
        nn.init.xavier_uniform_(se.mlp[2].weight)

        s1 = torch.tensor([[0.5, 0.5, 1.0]])
        s2 = torch.tensor([[1.5, 1.5, 5.0]])
        out1 = se(s1)
        out2 = se(s2)
        # They should differ
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_gradient_flow(self):
        """Gradients should flow through ScaleEmbedding."""
        se = ScaleEmbedding(embed_dim=64)
        # Re-init output to non-zero so gradient is meaningful
        nn.init.xavier_uniform_(se.mlp[2].weight)

        spacing = torch.tensor([[0.5, 0.5, 1.0]], requires_grad=True)
        out = se(spacing)
        loss = out.sum()
        loss.backward()
        assert spacing.grad is not None
        assert spacing.grad.abs().sum() > 0

    def test_small_embed_dim(self):
        """Should work with small embed_dim (hidden = max(dim//4, 16))."""
        se = ScaleEmbedding(embed_dim=16)
        spacing = torch.randn(2, 3)
        out = se(spacing)
        assert out.shape == (2, 1, 16)

    def test_large_embed_dim(self):
        """Should work with ViT-Large embedding dim."""
        se = ScaleEmbedding(embed_dim=1024)
        spacing = torch.randn(4, 3)
        out = se(spacing)
        assert out.shape == (4, 1, 1024)


class TestPatchViTScaleAware:
    """Test PatchViT with scale_aware flag."""

    def _make_vit(self, scale_aware: bool = False) -> PatchViT:
        return PatchViT(
            img_size=56,
            patch=14,
            dim=64,
            depth=2,
            heads=2,
            mlp_ratio=2.0,
            num_registers=2,
            scale_aware=scale_aware,
        )

    def test_backward_compat_no_scale(self):
        """Without scale_aware, ViT should work exactly as before."""
        vit = self._make_vit(scale_aware=False)
        x = torch.randn(2, 3, 56, 56)
        out = vit(x)
        # (2, 1 + 16 + 2, 64) = (2, 19, 64)
        n_patches = (56 // 14) ** 2  # = 16
        expected_seq = 1 + n_patches + 2  # CLS + patches + registers
        assert out.shape == (2, expected_seq, 64)

    def test_scale_aware_without_spacing(self):
        """Scale-aware ViT should still work when spacing=None (no-op)."""
        vit = self._make_vit(scale_aware=True)
        x = torch.randn(2, 3, 56, 56)
        out = vit(x)
        n_patches = (56 // 14) ** 2
        expected_seq = 1 + n_patches + 2
        assert out.shape == (2, expected_seq, 64)

    def test_scale_aware_with_spacing(self):
        """Scale-aware ViT should accept spacing and produce same shape."""
        vit = self._make_vit(scale_aware=True)
        x = torch.randn(2, 3, 56, 56)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 5.0]])
        out = vit(x, spacing=spacing)
        n_patches = (56 // 14) ** 2
        expected_seq = 1 + n_patches + 2
        assert out.shape == (2, expected_seq, 64)

    def test_zero_init_identity(self):
        """With zero-init'd scale embedding, output should be nearly identical to no-spacing."""
        vit = self._make_vit(scale_aware=True)
        x = torch.randn(2, 3, 56, 56)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 5.0]])

        # Fix randomness
        torch.manual_seed(42)
        out_none = vit(x)

        torch.manual_seed(42)
        out_spaced = vit(x, spacing=spacing)

        # Should be extremely close due to zero-init
        assert torch.allclose(out_none, out_spaced, atol=1e-5), \
            f"Zero-init should make spacing a no-op, max diff={((out_none - out_spaced).abs().max())}"

    def test_has_scale_embed_attribute(self):
        """Scale-aware ViT should have scale_embed; non-scale-aware should not."""
        vit_sa = self._make_vit(scale_aware=True)
        vit_no = self._make_vit(scale_aware=False)
        assert hasattr(vit_sa, "scale_embed")
        assert not hasattr(vit_no, "scale_embed")

    def test_scale_embed_param_count(self):
        """ScaleEmbedding should add a small number of parameters."""
        vit_no = self._make_vit(scale_aware=False)
        vit_sa = self._make_vit(scale_aware=True)
        params_no = sum(p.numel() for p in vit_no.parameters())
        params_sa = sum(p.numel() for p in vit_sa.parameters())
        # ScaleEmbedding: Linear(3, 16) + Linear(16, 64) + LayerNorm(64)
        # = (3*16+16) + (16*64+64) + (64+64) = 64 + 1088 + 128 = 1280
        extra = params_sa - params_no
        assert extra > 0, "Scale-aware should have more params"
        assert extra < 5000, f"Scale embed should be lightweight, got {extra} extra params"


class TestDinoStudentTeacherSpacing:
    """Test DinoStudentTeacher spacing passthrough."""

    def test_forward_without_spacing(self):
        vit = PatchViT(img_size=56, patch=14, dim=64, depth=2, heads=2, scale_aware=False)
        model = DinoStudentTeacher(vit, out_dim=128)
        x = torch.randn(2, 3, 56, 56)
        out = model(x)
        assert out.shape == (2, 128)

    def test_forward_with_spacing(self):
        vit = PatchViT(img_size=56, patch=14, dim=64, depth=2, heads=2, scale_aware=True)
        model = DinoStudentTeacher(vit, out_dim=128)
        x = torch.randn(2, 3, 56, 56)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 5.0]])
        out = model(x, spacing=spacing)
        assert out.shape == (2, 128)


class TestIndexRowSpacing:
    """Test IndexRow spacing fields and CSV loading."""

    def test_default_spacing(self):
        row = IndexRow(
            png_path=Path("test.png"),
            series_dir="s1",
            slice_index=0,
            encoding="HU",
        )
        assert row.spacing_x == 1.0
        assert row.spacing_y == 1.0
        assert row.spacing_z == 1.0

    def test_custom_spacing(self):
        row = IndexRow(
            png_path=Path("test.png"),
            series_dir="s1",
            slice_index=0,
            encoding="HU",
            spacing_x=0.5,
            spacing_y=0.5,
            spacing_z=1.25,
        )
        assert row.spacing_x == 0.5
        assert row.spacing_z == 1.25

    def test_load_csv_without_spacing(self):
        """CSV without spacing columns → defaults to 1.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["png_path", "series_dir", "slice_index", "encoding"])
            writer.writeheader()
            writer.writerow({"png_path": "a.png", "series_dir": "s1", "slice_index": "0", "encoding": "HU"})
            f.flush()
            rows = _load_index_rows(Path(f.name))
        assert len(rows) == 1
        assert rows[0].spacing_x == 1.0
        assert rows[0].spacing_y == 1.0
        assert rows[0].spacing_z == 1.0

    def test_load_csv_with_spacing(self):
        """CSV with spacing columns → values are read."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(
                f, fieldnames=["png_path", "series_dir", "slice_index", "encoding",
                               "spacing_x", "spacing_y", "spacing_z"]
            )
            writer.writeheader()
            writer.writerow({
                "png_path": "a.png", "series_dir": "s1", "slice_index": "0",
                "encoding": "HU", "spacing_x": "0.625", "spacing_y": "0.625", "spacing_z": "2.5",
            })
            f.flush()
            rows = _load_index_rows(Path(f.name))
        assert len(rows) == 1
        assert abs(rows[0].spacing_x - 0.625) < 1e-6
        assert abs(rows[0].spacing_z - 2.5) < 1e-6

    def test_load_csv_with_require_spacing_warns(self):
        """require_spacing=True warns when spacing columns are missing."""
        import warnings
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["png_path", "series_dir", "slice_index", "encoding"])
            writer.writeheader()
            writer.writerow({"png_path": "a.png", "series_dir": "s1", "slice_index": "0", "encoding": "HU"})
            f.flush()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rows = _load_index_rows(Path(f.name), require_spacing=True)
                assert len(w) == 1
                assert "scale-aware" in str(w[0].message).lower()
        assert rows[0].spacing_x == 1.0


class TestDinoCollate:
    """Test the dino_collate function."""

    def test_basic_collate(self):
        """Collate should stack views and spacings."""
        batch = [
            ([torch.randn(3, 56, 56), torch.randn(3, 56, 56)], torch.tensor([0.5, 0.5, 1.0])),
            ([torch.randn(3, 56, 56), torch.randn(3, 56, 56)], torch.tensor([1.5, 1.5, 5.0])),
        ]
        views, spacing = dino_collate(batch)
        assert len(views) == 2
        assert views[0].shape == (2, 3, 56, 56)
        assert views[1].shape == (2, 3, 56, 56)
        assert spacing.shape == (2, 3)

    def test_spacing_values_preserved(self):
        """Spacing values should be preserved through collation."""
        s1 = torch.tensor([0.5, 0.5, 1.0])
        s2 = torch.tensor([1.5, 1.5, 5.0])
        batch = [
            ([torch.zeros(3, 56, 56), torch.zeros(3, 56, 56)], s1),
            ([torch.zeros(3, 56, 56), torch.zeros(3, 56, 56)], s2),
        ]
        _, spacing = dino_collate(batch)
        assert torch.allclose(spacing[0], s1)
        assert torch.allclose(spacing[1], s2)


class TestEndToEnd:
    """Integration test: full forward pass with scale embedding."""

    def test_dino_forward_with_scale(self):
        """Student and teacher forward passes with spacing should produce correct shapes."""
        vit_s = PatchViT(img_size=56, patch=14, dim=64, depth=2, heads=2, scale_aware=True)
        student = DinoStudentTeacher(vit_s, out_dim=128)

        vit_t = PatchViT(img_size=56, patch=14, dim=64, depth=2, heads=2, scale_aware=True)
        teacher = DinoStudentTeacher(vit_t, out_dim=128)
        teacher.load_state_dict(student.state_dict())

        B = 4
        x = torch.randn(2 * B, 3, 56, 56)
        spacing = torch.randn(2 * B, 3).abs()  # spacing should be positive

        s_out = student(x, spacing=spacing)
        with torch.no_grad():
            t_out = teacher(x, spacing=spacing)

        assert s_out.shape == (2 * B, 128)
        assert t_out.shape == (2 * B, 128)

    def test_backward_pass_with_scale(self):
        """Loss backward should work and update scale embedding weights."""
        vit = PatchViT(img_size=56, patch=14, dim=64, depth=2, heads=2, scale_aware=True)
        # Un-zero the output layer so gradients flow meaningfully
        nn.init.xavier_uniform_(vit.scale_embed.mlp[2].weight)

        model = DinoStudentTeacher(vit, out_dim=128)
        x = torch.randn(2, 3, 56, 56)
        spacing = torch.tensor([[0.5, 0.5, 1.0], [1.5, 1.5, 5.0]])

        out = model(x, spacing=spacing)
        loss = out.sum()
        loss.backward()

        # Check scale embedding gradients exist
        for name, p in model.named_parameters():
            if "scale_embed" in name and p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

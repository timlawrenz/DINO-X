"""Tests for scripts/finetune_lora.py — LoRA fine-tuning pipeline."""

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zoo.arch import PatchViT, DinoStudentTeacher

# Import the fine-tuning module
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from finetune_lora import (
    LabeledImageDataset,
    LabeledRow,
    FinetuneModel,
    _load_csv,
    compute_metrics,
    FinetuneConfig,
    save_finetune,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_backbone() -> PatchViT:
    """A minimal PatchViT for fast testing."""
    return PatchViT(
        img_size=32, patch=8, dim=64, depth=2, heads=2,
        mlp_ratio=2.0, num_registers=0, scale_aware=False,
    )


@pytest.fixture
def tiny_scale_backbone() -> PatchViT:
    """A minimal scale-aware PatchViT."""
    return PatchViT(
        img_size=32, patch=8, dim=64, depth=2, heads=2,
        mlp_ratio=2.0, num_registers=0, scale_aware=True,
    )


@pytest.fixture
def sample_images(tmp_path: Path) -> Path:
    """Create a few synthetic 16-bit PNG images."""
    from PIL import Image

    for i in range(10):
        arr = np.random.randint(30000, 35000, (64, 64), dtype=np.uint16)
        img = Image.fromarray(arr)
        img.save(tmp_path / f"img_{i:03d}.png")
    return tmp_path


@pytest.fixture
def train_csv(tmp_path: Path, sample_images: Path) -> Path:
    """Create a training CSV with 8 samples."""
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for i in range(8):
            writer.writerow({
                "image_path": str(sample_images / f"img_{i:03d}.png"),
                "label": str(i % 3),
            })
    return csv_path


@pytest.fixture
def val_csv(tmp_path: Path, sample_images: Path) -> Path:
    """Create a validation CSV with 2 samples."""
    csv_path = tmp_path / "val.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for i in range(8, 10):
            writer.writerow({
                "image_path": str(sample_images / f"img_{i:03d}.png"),
                "label": str(i % 3),
            })
    return csv_path


@pytest.fixture
def train_csv_with_spacing(tmp_path: Path, sample_images: Path) -> Path:
    """Create a training CSV with spacing columns."""
    csv_path = tmp_path / "train_spacing.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "label", "spacing_x", "spacing_y", "spacing_z"]
        )
        writer.writeheader()
        for i in range(8):
            writer.writerow({
                "image_path": str(sample_images / f"img_{i:03d}.png"),
                "label": str(i % 3),
                "spacing_x": "0.7",
                "spacing_y": "0.7",
                "spacing_z": "1.5",
            })
    return csv_path


@pytest.fixture
def regression_csv(tmp_path: Path, sample_images: Path) -> Path:
    """Create a regression CSV."""
    csv_path = tmp_path / "regression.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        for i in range(8):
            writer.writerow({
                "image_path": str(sample_images / f"img_{i:03d}.png"),
                "label": f"{i * 0.5:.1f}",
            })
    return csv_path


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_basic_csv(self, train_csv: Path) -> None:
        rows = _load_csv(train_csv)
        assert len(rows) == 8
        assert all(isinstance(r, LabeledRow) for r in rows)
        assert rows[0].label in (0.0, 1.0, 2.0)
        assert not rows[0].has_spacing

    def test_csv_with_spacing(self, train_csv_with_spacing: Path) -> None:
        rows = _load_csv(train_csv_with_spacing)
        assert len(rows) == 8
        assert rows[0].has_spacing
        assert rows[0].spacing_x == 0.7
        assert rows[0].spacing_y == 0.7
        assert rows[0].spacing_z == 1.5

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w") as f:
            f.write("image_path,score\n")
            f.write("img.png,1\n")

        with pytest.raises(ValueError, match="missing required columns"):
            _load_csv(csv_path)

    def test_default_spacing(self, train_csv: Path) -> None:
        rows = _load_csv(train_csv)
        assert rows[0].spacing_x == 1.0
        assert rows[0].spacing_y == 1.0
        assert rows[0].spacing_z == 1.0


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestLabeledImageDataset:
    def test_len(self, train_csv: Path) -> None:
        rows = _load_csv(train_csv)
        ds = LabeledImageDataset(rows, img_size=32, input_format="hu16_png")
        assert len(ds) == 8

    def test_getitem_shape(self, train_csv: Path) -> None:
        rows = _load_csv(train_csv)
        ds = LabeledImageDataset(rows, img_size=32, input_format="hu16_png")
        x, spacing, label = ds[0]
        assert x.shape == (3, 32, 32)
        assert spacing.shape == (3,)
        assert isinstance(label, float)

    def test_augment_mode(self, train_csv: Path) -> None:
        rows = _load_csv(train_csv)
        ds = LabeledImageDataset(
            rows, img_size=32, input_format="hu16_png", augment=True,
        )
        x, _, _ = ds[0]
        assert x.shape == (3, 32, 32)

    def test_data_root(self, tmp_path: Path, sample_images: Path) -> None:
        csv_path = tmp_path / "rel.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
            writer.writeheader()
            writer.writerow({"image_path": "img_000.png", "label": "0"})

        rows = _load_csv(csv_path)
        ds = LabeledImageDataset(
            rows, img_size=32, input_format="hu16_png",
            data_root=sample_images,
        )
        x, _, _ = ds[0]
        assert x.shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestFinetuneModel:
    def test_classification_forward(self, tiny_backbone: PatchViT) -> None:
        model = FinetuneModel(tiny_backbone, dim=64, num_classes=3)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 3)

    def test_regression_forward(self, tiny_backbone: PatchViT) -> None:
        model = FinetuneModel(tiny_backbone, dim=64, num_classes=1, task="regression")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 1)

    def test_scale_aware_forward(self, tiny_scale_backbone: PatchViT) -> None:
        model = FinetuneModel(tiny_scale_backbone, dim=64, num_classes=3)
        x = torch.randn(2, 3, 32, 32)
        spacing = torch.tensor([[0.7, 0.7, 1.5], [1.0, 1.0, 3.0]])
        logits = model(x, spacing=spacing)
        assert logits.shape == (2, 3)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_classification_perfect(self) -> None:
        preds = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([0, 1])
        m = compute_metrics(preds, targets, "classification", 2)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == pytest.approx(1.0, abs=0.01)

    def test_classification_wrong(self) -> None:
        preds = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
        targets = torch.tensor([0, 1])
        m = compute_metrics(preds, targets, "classification", 2)
        assert m["accuracy"] == 0.0

    def test_regression_perfect(self) -> None:
        preds = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([1.0, 2.0, 3.0])
        m = compute_metrics(preds, targets, "regression", 1)
        assert m["mse"] == pytest.approx(0.0, abs=1e-6)
        assert m["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_regression_imperfect(self) -> None:
        preds = torch.tensor([[1.5], [2.5], [3.5]])
        targets = torch.tensor([1.0, 2.0, 3.0])
        m = compute_metrics(preds, targets, "regression", 1)
        assert m["mse"] > 0
        assert m["r2"] < 1.0

    def test_multiclass_f1(self) -> None:
        preds = torch.tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 0, 0]])
        targets = torch.tensor([0, 1, 2, 0])
        m = compute_metrics(preds, targets, "classification", 3)
        assert m["macro_f1"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Save tests
# ---------------------------------------------------------------------------


class TestSaveFinetune:
    def test_save_creates_files(self, tiny_backbone: PatchViT, tmp_path: Path) -> None:
        from zoo.peft import apply_lora

        backbone = apply_lora(tiny_backbone, rank=4, alpha=8.0)
        model = FinetuneModel(backbone, dim=64, num_classes=3)

        config = FinetuneConfig(
            backbone="test", task="classification", num_classes=3,
            rank=4, alpha=8.0, lr=1e-3, epochs=10, batch_size=32,
            input_format="hu16_png", scale_aware=False,
        )

        out = tmp_path / "adapter_out"
        save_finetune(model, out, config)

        assert (out / "head.pth").exists()
        assert (out / "finetune_config.json").exists()
        # peft saves adapter_config.json and adapter_model files
        assert (out / "adapter_config.json").exists()

        # Check config is valid JSON
        cfg = json.loads((out / "finetune_config.json").read_text())
        assert cfg["task"] == "classification"
        assert cfg["rank"] == 4

    def test_head_weights_loadable(self, tiny_backbone: PatchViT, tmp_path: Path) -> None:
        from zoo.peft import apply_lora

        backbone = apply_lora(tiny_backbone, rank=4, alpha=8.0)
        model = FinetuneModel(backbone, dim=64, num_classes=3)

        out = tmp_path / "adapter_out2"
        config = FinetuneConfig(
            backbone="test", task="classification", num_classes=3,
            rank=4, alpha=8.0, lr=1e-3, epochs=10, batch_size=32,
            input_format="hu16_png", scale_aware=False,
        )
        save_finetune(model, out, config)

        # Load head weights into a fresh head
        new_head = torch.nn.Linear(64, 3)
        new_head.load_state_dict(torch.load(out / "head.pth", weights_only=True))
        assert new_head.weight.shape == (3, 64)

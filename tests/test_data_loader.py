"""Tests for zoo/data.py — Unified CT data loader."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zoo.data import ManifestDataset, ManifestEvalDataset, dino_collate
from zoo.manifest import DataManifest
from zoo.models import SliceMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hu16_png(path: Path, value: int = 33000) -> None:
    """Create a synthetic 16-bit HU PNG."""
    arr = np.full((64, 64), value, dtype=np.uint16)
    Image.fromarray(arr).save(path)


@pytest.fixture
def sample_data(tmp_path: Path) -> tuple[DataManifest, Path]:
    """Create a manifest with 2 datasets, 2 series each, 5 slices each."""
    data_root = tmp_path / "data"
    records: list[SliceMetadata] = []

    for ds_name, organ in [("lidc-idri", "lung"), ("pancreas-ct", "pancreas")]:
        for series_idx in range(2):
            series_id = f"series_{ds_name}_{series_idx}"
            series_dir = data_root / ds_name / series_id
            series_dir.mkdir(parents=True)

            for z in range(5):
                img_path = series_dir / f"slice_{z:03d}.png"
                _make_hu16_png(img_path)

                records.append(SliceMetadata(
                    dataset=ds_name,
                    series_id=series_id,
                    slice_idx=z,
                    pixel_spacing_x=0.7 if ds_name == "lidc-idri" else 1.2,
                    pixel_spacing_y=0.7 if ds_name == "lidc-idri" else 1.2,
                    slice_thickness=1.5 if ds_name == "lidc-idri" else 3.0,
                    image_path=str(img_path),
                    organs_present=[organ],
                ))

    manifest = DataManifest(records)
    return manifest, data_root


@pytest.fixture
def relative_data(tmp_path: Path) -> tuple[DataManifest, Path]:
    """Create a manifest with relative paths."""
    data_root = tmp_path / "relative_data"
    data_root.mkdir()

    records: list[SliceMetadata] = []
    for z in range(3):
        img_path = data_root / f"slice_{z:03d}.png"
        _make_hu16_png(img_path)

        records.append(SliceMetadata(
            dataset="test",
            series_id="s0",
            slice_idx=z,
            pixel_spacing_x=0.5,
            pixel_spacing_y=0.5,
            slice_thickness=1.0,
            image_path=f"slice_{z:03d}.png",
            organs_present=["lung"],
        ))

    manifest = DataManifest(records)
    return manifest, data_root


# ---------------------------------------------------------------------------
# ManifestDataset tests
# ---------------------------------------------------------------------------


class TestManifestDataset:
    def test_len(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)
        assert len(ds) == 20  # 2 datasets × 2 series × 5 slices

    def test_getitem_shapes(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)
        views, spacing = ds[0]
        assert len(views) == 2
        assert views[0].shape == (3, 32, 32)
        assert views[1].shape == (3, 32, 32)
        assert spacing.shape == (3,)

    def test_spacing_values(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)

        # First 10 records are lidc-idri
        _, spacing = ds[0]
        assert spacing[0].item() == pytest.approx(0.7)
        assert spacing[2].item() == pytest.approx(1.5)

        # Record 10+ are pancreas-ct
        _, spacing = ds[10]
        assert spacing[0].item() == pytest.approx(1.2)
        assert spacing[2].item() == pytest.approx(3.0)

    def test_two_views_differ(self, sample_data) -> None:
        """DINO needs two different augmented views of the same image."""
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)
        views, _ = ds[0]
        # Views should differ due to random windowing + augmentation
        assert not torch.allclose(views[0], views[1])

    def test_relative_paths_with_data_root(self, relative_data) -> None:
        manifest, data_root = relative_data
        ds = ManifestDataset.from_manifest(
            manifest, img_size=32, data_root=data_root,
        )
        views, spacing = ds[0]
        assert views[0].shape == (3, 32, 32)

    def test_from_parquet_roundtrip(self, sample_data, tmp_path: Path) -> None:
        manifest, _ = sample_data
        pq_path = tmp_path / "manifest.parquet"
        manifest.save(pq_path)

        ds = ManifestDataset.from_parquet(pq_path, img_size=32)
        assert len(ds) == 20
        views, _ = ds[0]
        assert views[0].shape == (3, 32, 32)

    def test_context_boundary_clamping(self, sample_data) -> None:
        """First and last slices should use boundary clamping."""
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)

        # Index 0 is slice_idx=0 (first in series) — z-1 should clamp to z=0
        views, _ = ds[0]
        assert views[0].shape == (3, 32, 32)  # Should not error

        # Index 4 is slice_idx=4 (last in series) — z+1 should clamp to z=4
        views, _ = ds[4]
        assert views[0].shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# ManifestEvalDataset tests
# ---------------------------------------------------------------------------


class TestManifestEvalDataset:
    def test_len(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestEvalDataset.from_manifest(manifest, img_size=32)
        assert len(ds) == 20

    def test_getitem_shapes(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestEvalDataset.from_manifest(manifest, img_size=32)
        image, spacing = ds[0]
        assert image.shape == (3, 32, 32)
        assert spacing.shape == (3,)

    def test_deterministic(self, sample_data) -> None:
        """Eval dataset should produce identical outputs for same index."""
        manifest, _ = sample_data
        ds = ManifestEvalDataset.from_manifest(manifest, img_size=32)
        img1, sp1 = ds[0]
        img2, sp2 = ds[0]
        assert torch.allclose(img1, img2)
        assert torch.allclose(sp1, sp2)

    def test_from_parquet(self, sample_data, tmp_path: Path) -> None:
        manifest, _ = sample_data
        pq_path = tmp_path / "eval_manifest.parquet"
        manifest.save(pq_path)

        ds = ManifestEvalDataset.from_parquet(pq_path, img_size=32)
        assert len(ds) == 20


# ---------------------------------------------------------------------------
# Collate tests
# ---------------------------------------------------------------------------


class TestDinoCollate:
    def test_collate_shapes(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)

        batch = [ds[i] for i in range(4)]
        views, spacing = dino_collate(batch)

        assert len(views) == 2
        assert views[0].shape == (4, 3, 32, 32)
        assert views[1].shape == (4, 3, 32, 32)
        assert spacing.shape == (4, 3)

    def test_collate_preserves_spacing(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)

        # Mix records from different datasets
        batch = [ds[0], ds[10]]  # lidc + pancreas
        views, spacing = dino_collate(batch)

        assert spacing[0, 0].item() == pytest.approx(0.7)
        assert spacing[1, 0].item() == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# DataLoader integration tests
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_training_dataloader(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestDataset.from_manifest(manifest, img_size=32)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=dino_collate, num_workers=0,
        )

        batch = next(iter(loader))
        views, spacing = batch
        assert views[0].shape == (4, 3, 32, 32)
        assert spacing.shape == (4, 3)

    def test_eval_dataloader(self, sample_data) -> None:
        manifest, _ = sample_data
        ds = ManifestEvalDataset.from_manifest(manifest, img_size=32)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, num_workers=0,
        )

        images, spacings = next(iter(loader))
        assert images.shape == (4, 3, 32, 32)
        assert spacings.shape == (4, 3)

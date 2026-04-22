"""Unified CT data loader reading from DataManifest Parquet files.

Bridges the zoo data registry (Parquet manifests with per-slice
physical spacing) to the PyTorch training pipeline. Supports
multi-dataset interleaving, 3-slice context, random HU windowing,
and scale-aware spacing tensors.

Usage::

    from zoo.data import ManifestDataset, dino_collate

    ds = ManifestDataset.from_parquet(
        "data/combined_manifest.parquet",
        img_size=224,
        scale_aware=True,
    )
    loader = DataLoader(ds, batch_size=64, collate_fn=dino_collate)

    for views, spacing in loader:
        # views: [view1_batch, view2_batch], each (B, 3, H, W)
        # spacing: (B, 3)
        ...
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from zoo.manifest import DataManifest
from zoo.models import SliceMetadata

logger = logging.getLogger(__name__)


class ManifestDataset(torch.utils.data.Dataset):
    """Training dataset backed by a DataManifest with physical spacing.

    Each sample provides two randomly-windowed views (for DINO cross-view
    prediction) and a spacing tensor for the ScaleEmbedding.

    Features:
        - Reads from Parquet manifest (SliceMetadata records)
        - 3-slice context (z-1, z, z+1) via series grouping
        - Random HU windowing augmentation
        - Spatial augmentations (RandomResizedCrop + HorizontalFlip)
        - Scale-aware spacing tensors from manifest metadata
    """

    def __init__(
        self,
        records: list[SliceMetadata],
        img_size: int = 224,
        rw_level_range: tuple[float, float] = (-400.0, 400.0),
        rw_width_range: tuple[float, float] = (800.0, 2000.0),
        scale_aware: bool = False,
        data_root: Path | str | None = None,
    ):
        """Initialize from a list of SliceMetadata records.

        Args:
            records: Per-slice metadata from a DataManifest.
            img_size: Target image size after crop.
            rw_level_range: (min, max) for random window level (HU).
            rw_width_range: (min, max) for random window width (HU).
            scale_aware: Whether to provide spacing tensors.
            data_root: Root directory for resolving relative image paths.
        """
        self.records = records
        self.img_size = img_size
        self.rw_level_min, self.rw_level_max = rw_level_range
        self.rw_width_min, self.rw_width_max = rw_width_range
        self.scale_aware = scale_aware
        self.data_root = Path(data_root) if data_root else None

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Build series index for 3-slice context lookup
        self._series_map: dict[str, dict[int, str]] = {}
        self._series_bounds: dict[str, tuple[int, int]] = {}

        for r in records:
            key = f"{r.dataset}:{r.series_id}"
            sm = self._series_map.setdefault(key, {})
            sm[r.slice_idx] = r.image_path

        for key, sm in self._series_map.items():
            if sm:
                ks = sorted(sm.keys())
                self._series_bounds[key] = (ks[0], ks[-1])

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        **kwargs,
    ) -> ManifestDataset:
        """Load dataset from a Parquet manifest file.

        Args:
            path: Path to Parquet manifest.
            **kwargs: Passed to ``ManifestDataset.__init__``.
        """
        manifest = DataManifest.load(path)
        return cls(manifest.records, **kwargs)

    @classmethod
    def from_manifest(
        cls,
        manifest: DataManifest,
        **kwargs,
    ) -> ManifestDataset:
        """Create dataset from an in-memory DataManifest.

        Args:
            manifest: A DataManifest instance.
            **kwargs: Passed to ``ManifestDataset.__init__``.
        """
        return cls(manifest.records, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    def _load_hu01(
        self, path: Path, level: float, width: float,
    ) -> np.ndarray:
        """Load a 16-bit HU PNG and apply windowing to [0, 1]."""
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        hu = (arr - 32768.0) * 0.1
        wmin = level - width / 2.0
        windowed = (hu - wmin) / max(width, 1.0)
        return np.clip(windowed, 0.0, 1.0)

    def _get_context_paths(self, record: SliceMetadata) -> list[str]:
        """Get paths for z-1, z, z+1 slices (clamped to series bounds)."""
        key = f"{record.dataset}:{record.series_id}"
        sm = self._series_map.get(key, {})
        z = record.slice_idx
        z0, z1 = self._series_bounds.get(key, (z, z))

        def clamp(k: int) -> int:
            return max(z0, min(z1, k))

        return [
            sm.get(clamp(z - 1), record.image_path),
            sm.get(clamp(z), record.image_path),
            sm.get(clamp(z + 1), record.image_path),
        ]

    def __getitem__(
        self, idx: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return (views, spacing) for one sample.

        Returns:
            views: List of 2 tensors, each (3, H, W) — two augmented views.
            spacing: (3,) tensor of (spacing_x, spacing_y, spacing_z).
        """
        attempts = 0
        while attempts < 10:
            try:
                record = self.records[idx]
                context_paths = self._get_context_paths(record)
                resolved = [self._resolve_path(p) for p in context_paths]

                def _get_view() -> torch.Tensor:
                    level = random.uniform(self.rw_level_min, self.rw_level_max)
                    width = random.uniform(self.rw_width_min, self.rw_width_max)
                    slices = [self._load_hu01(p, level, width) for p in resolved]
                    x = np.stack(slices, axis=0)  # (3, H, W)
                    return self.transform(torch.from_numpy(x).contiguous())

                spacing = torch.tensor(
                    [record.pixel_spacing_x, record.pixel_spacing_y,
                     record.slice_thickness],
                    dtype=torch.float32,
                )

                return [_get_view(), _get_view()], spacing

            except Exception as e:
                logger.warning(
                    "Data loading error at index %d (%s): %s",
                    idx, self.records[idx].image_path, e,
                )
                idx = random.randint(0, len(self.records) - 1)
                attempts += 1

        raise RuntimeError("Failed to load data after 10 attempts")


class ManifestEvalDataset(torch.utils.data.Dataset):
    """Deterministic evaluation dataset from a DataManifest.

    Uses fixed windowing (no random augmentation) for reproducible
    evaluation. Returns single views instead of pairs.
    """

    def __init__(
        self,
        records: list[SliceMetadata],
        img_size: int = 224,
        window_level: float = 40.0,
        window_width: float = 400.0,
        data_root: Path | str | None = None,
    ):
        self.records = records
        self.img_size = img_size
        self.window_level = window_level
        self.window_width = window_width
        self.data_root = Path(data_root) if data_root else None

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

        # Series index for 3-slice context
        self._series_map: dict[str, dict[int, str]] = {}
        self._series_bounds: dict[str, tuple[int, int]] = {}

        for r in records:
            key = f"{r.dataset}:{r.series_id}"
            sm = self._series_map.setdefault(key, {})
            sm[r.slice_idx] = r.image_path

        for key, sm in self._series_map.items():
            if sm:
                ks = sorted(sm.keys())
                self._series_bounds[key] = (ks[0], ks[-1])

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs) -> ManifestEvalDataset:
        manifest = DataManifest.load(path)
        return cls(manifest.records, **kwargs)

    @classmethod
    def from_manifest(cls, manifest: DataManifest, **kwargs) -> ManifestEvalDataset:
        return cls(manifest.records, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, image_path: str) -> Path:
        p = Path(image_path)
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    def _load_hu01(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        hu = (arr - 32768.0) * 0.1
        wmin = self.window_level - self.window_width / 2.0
        windowed = (hu - wmin) / max(self.window_width, 1.0)
        return np.clip(windowed, 0.0, 1.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, spacing) for one sample."""
        record = self.records[idx]
        key = f"{record.dataset}:{record.series_id}"
        sm = self._series_map.get(key, {})
        z = record.slice_idx
        z0, z1 = self._series_bounds.get(key, (z, z))

        def clamp(k: int) -> int:
            return max(z0, min(z1, k))

        paths = [
            sm.get(clamp(z - 1), record.image_path),
            sm.get(clamp(z), record.image_path),
            sm.get(clamp(z + 1), record.image_path),
        ]
        resolved = [self._resolve_path(p) for p in paths]
        slices = [self._load_hu01(p) for p in resolved]
        x = np.stack(slices, axis=0)
        x = self.transform(torch.from_numpy(x).contiguous())

        spacing = torch.tensor(
            [record.pixel_spacing_x, record.pixel_spacing_y,
             record.slice_thickness],
            dtype=torch.float32,
        )
        return x, spacing


def dino_collate(
    batch: list[tuple[list[torch.Tensor], torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Custom collate for ManifestDataset (DINO training).

    Args:
        batch: List of (views_list, spacing_tensor) tuples.

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

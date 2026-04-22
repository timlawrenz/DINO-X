"""Tests for evaluate_panorgan.py metric functions.

Uses synthetic data to verify metric computation logic without
requiring GPU, checkpoint files, or real medical images.
"""

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Minimal stubs so we can import evaluate_panorgan metric functions without
# loading phase5_big_run (which requires torch, PIL, etc.)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeRow:
    """Minimal stand-in for phase5_big_run.IndexRow used by metric functions."""
    png_path: Path = Path("fake.png")
    series_dir: str = "series_0"
    slice_index: int = 0
    encoding: str = "hu16"
    spacing_x: float = 1.0
    spacing_y: float = 1.0
    spacing_z: float = 1.0
    dataset: str = ""


def _make_rows(
    n_ds1: int = 100,
    n_ds2: int = 80,
    n_series_ds1: int = 5,
    n_series_ds2: int = 4,
    spacing_ds1: float = 0.5,
    spacing_ds2: float = 1.5,
) -> list[FakeRow]:
    """Create synthetic rows with two datasets and distinct spacings."""
    rows = []
    slices_per_series_1 = n_ds1 // n_series_ds1
    slices_per_series_2 = n_ds2 // n_series_ds2

    for s in range(n_series_ds1):
        for z in range(slices_per_series_1):
            rows.append(FakeRow(
                series_dir=f"lung_series_{s}",
                slice_index=z,
                spacing_x=spacing_ds1 + random.gauss(0, 0.05),
                spacing_y=spacing_ds1 + random.gauss(0, 0.05),
                spacing_z=1.0,
                dataset="lidc-idri",
            ))

    for s in range(n_series_ds2):
        for z in range(slices_per_series_2):
            rows.append(FakeRow(
                series_dir=f"pancreas_series_{s}",
                slice_index=z,
                spacing_x=spacing_ds2 + random.gauss(0, 0.05),
                spacing_y=spacing_ds2 + random.gauss(0, 0.05),
                spacing_z=2.5,
                dataset="pancreas-ct",
            ))

    return rows


def _make_clustered_embeddings(rows: list[FakeRow], dim: int = 384, sep: float = 2.0) -> np.ndarray:
    """Create embeddings that cluster by dataset (simulates a trained model)."""
    rng = np.random.RandomState(42)
    embs = np.zeros((len(rows), dim), dtype=np.float32)

    for i, r in enumerate(rows):
        if r.dataset == "lidc-idri":
            center = np.zeros(dim)
            center[0] = sep
        else:
            center = np.zeros(dim)
            center[0] = -sep
        embs[i] = center + rng.randn(dim) * 0.3

    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-8)
    return embs


def _make_random_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Create random (unclustered) embeddings."""
    rng = np.random.RandomState(42)
    embs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Import the metric functions. They only need numpy + sklearn, not torch.
# We import them by adding scripts/ to path.
# ─────────────────────────────────────────────────────────────────────────────

# The metric functions we want to test operate on numpy arrays and FakeRow-like
# objects, so we can test them without importing the full module (which requires
# torch). Instead, we copy the pure-Python functions or mock imports.

# For a cleaner approach, let's directly test the logic by importing after
# stubbing out torch.
@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)


class TestDatasetDiscriminationProbe:
    """Test metric_dataset_discrimination_probe."""

    def test_perfect_clustering_gives_high_accuracy(self):
        """Well-separated embeddings → near-perfect probe accuracy."""
        from scripts.evaluate_panorgan import metric_dataset_discrimination_probe

        rows = _make_rows(n_ds1=200, n_ds2=160)
        embs = _make_clustered_embeddings(rows, sep=5.0)

        result = metric_dataset_discrimination_probe(embs, rows, seed=42)

        assert "accuracy" in result
        assert result["accuracy"] >= 0.9, f"Expected high accuracy, got {result['accuracy']}"
        assert result["auc"] >= 0.9
        assert result["train_series"] > 0
        assert result["test_series"] > 0

    def test_random_embeddings_give_low_accuracy(self):
        """Random embeddings → accuracy near chance (0.5 for 2 classes)."""
        from scripts.evaluate_panorgan import metric_dataset_discrimination_probe

        rows = _make_rows(n_ds1=200, n_ds2=200)
        embs = _make_random_embeddings(len(rows))

        result = metric_dataset_discrimination_probe(embs, rows, seed=42)

        assert "accuracy" in result
        # With random embeddings, accuracy should be near 0.5
        assert result["accuracy"] < 0.85, f"Expected low accuracy, got {result['accuracy']}"

    def test_series_level_split(self):
        """Verify probe uses series-level split (no slice leakage)."""
        from scripts.evaluate_panorgan import metric_dataset_discrimination_probe

        rows = _make_rows(n_ds1=100, n_ds2=80, n_series_ds1=5, n_series_ds2=4)
        embs = _make_clustered_embeddings(rows)

        result = metric_dataset_discrimination_probe(embs, rows, seed=42)

        # Total series = 5 + 4 = 9
        # Stratified: 80% per dataset → 4 lung train + 1 test, 3 pancreas train + 1 test
        assert result["train_series"] + result["test_series"] == 9
        assert result["test_series"] >= 2  # at least 1 per dataset

    def test_bootstrap_ci(self):
        """Verify bootstrap CI is computed and reasonable."""
        from scripts.evaluate_panorgan import metric_dataset_discrimination_probe

        rows = _make_rows(n_ds1=200, n_ds2=160)
        embs = _make_clustered_embeddings(rows, sep=3.0)

        result = metric_dataset_discrimination_probe(embs, rows, seed=42)

        assert "accuracy_ci95" in result
        ci = result["accuracy_ci95"]
        assert len(ci) == 2
        assert ci[0] <= result["accuracy"] <= ci[1] or abs(ci[0] - ci[1]) < 0.01


class TestDomainClustering:
    """Test metric_domain_clustering."""

    def test_clustered_embeddings_show_enrichment(self):
        """Well-clustered embeddings → enrichment > 1."""
        from scripts.evaluate_panorgan import metric_domain_clustering

        rows = _make_rows()
        embs = _make_clustered_embeddings(rows, sep=5.0)

        result = metric_domain_clustering(embs, rows, k=5)

        assert result["enrichment_vs_random"] > 1.5
        assert result["overall_same_dataset_rate"] > result["expected_random_rate"]

    def test_random_embeddings_near_baseline(self):
        """Random embeddings → enrichment ≈ 1."""
        from scripts.evaluate_panorgan import metric_domain_clustering

        rows = _make_rows(n_ds1=200, n_ds2=200)
        embs = _make_random_embeddings(len(rows))

        result = metric_domain_clustering(embs, rows, k=10)

        # Should be close to 1.0 (no clustering beyond random)
        assert 0.5 < result["enrichment_vs_random"] < 2.0

    def test_per_dataset_results(self):
        """Verify per-dataset breakdowns are present."""
        from scripts.evaluate_panorgan import metric_domain_clustering

        rows = _make_rows()
        embs = _make_clustered_embeddings(rows)

        result = metric_domain_clustering(embs, rows, k=5)

        assert "lidc-idri" in result["per_dataset"]
        assert "pancreas-ct" in result["per_dataset"]
        for ds in result["per_dataset"].values():
            assert "same_dataset_rate" in ds
            assert "expected_random" in ds
            assert "enrichment" in ds


class TestSpacingPrediction:
    """Test metric_spacing_prediction."""

    def test_spacing_encoded_in_embeddings(self):
        """If embeddings encode spacing, R² should be positive."""
        from scripts.evaluate_panorgan import metric_spacing_prediction

        rows = _make_rows(
            n_ds1=500, n_ds2=400,
            n_series_ds1=10, n_series_ds2=8,
            spacing_ds1=0.5, spacing_ds2=1.5,
        )
        # Make embeddings that correlate with spacing
        embs = _make_clustered_embeddings(rows, sep=5.0)
        spacings = np.array([[r.spacing_x, r.spacing_y, r.spacing_z] for r in rows], dtype=np.float32)

        result = metric_spacing_prediction(embs, spacings, rows, seed=42)

        assert "r2" in result
        # Clustered embeddings correlate with dataset which correlates with spacing
        assert result["r2"] > 0.0

    def test_random_embeddings_low_r2(self):
        """Random embeddings → R² near 0."""
        from scripts.evaluate_panorgan import metric_spacing_prediction

        rows = _make_rows(n_ds1=200, n_ds2=160)
        embs = _make_random_embeddings(len(rows))
        spacings = np.array([[r.spacing_x, r.spacing_y, r.spacing_z] for r in rows], dtype=np.float32)

        result = metric_spacing_prediction(embs, spacings, rows, seed=42)

        assert "r2" in result
        assert result["r2"] < 0.5


class TestEmbeddingStats:
    """Test metric_embedding_stats."""

    def test_basic_structure(self):
        """Verify output structure is correct."""
        from scripts.evaluate_panorgan import metric_embedding_stats

        rows = _make_rows()
        embs = _make_clustered_embeddings(rows)
        spacings = np.array([[r.spacing_x, r.spacing_y, r.spacing_z] for r in rows], dtype=np.float32)

        result = metric_embedding_stats(embs, spacings, rows)

        assert "per_dataset" in result
        assert "cross_dataset_centroid_cosine" in result
        assert "lidc-idri" in result["per_dataset"]
        assert "pancreas-ct" in result["per_dataset"]

        for ds_stats in result["per_dataset"].values():
            assert "embedding_std" in ds_stats
            assert "intra_cosine_to_centroid" in ds_stats
            assert "pca1_spacing_correlation" in ds_stats
            assert ds_stats["embedding_std"] > 0

    def test_clustered_cross_dataset_cosine(self):
        """Well-separated clusters → low cross-dataset centroid cosine."""
        from scripts.evaluate_panorgan import metric_embedding_stats

        rows = _make_rows()
        embs = _make_clustered_embeddings(rows, sep=10.0)
        spacings = np.array([[r.spacing_x, r.spacing_y, r.spacing_z] for r in rows], dtype=np.float32)

        result = metric_embedding_stats(embs, spacings, rows)

        cross = result["cross_dataset_centroid_cosine"]
        # With high separation, cross-dataset cosine should be negative
        assert "lidc-idri_vs_pancreas-ct" in cross
        assert cross["lidc-idri_vs_pancreas-ct"] < 0.5

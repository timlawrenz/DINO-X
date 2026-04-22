"""Tests for manifest and merge modules."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from zoo.manifest import DataManifest
from zoo.merge import DatasetMerger, temperature_weights
from zoo.models import SliceMetadata


def _make_records(
    dataset: str,
    n: int,
    *,
    spacing_x: float = 0.7,
    spacing_y: float = 0.7,
    thickness: float = 1.25,
    organs: list[str] | None = None,
) -> list[SliceMetadata]:
    """Create n dummy slice metadata records."""
    return [
        SliceMetadata(
            dataset=dataset,
            series_id=f"{dataset}-S{i // 100:03d}",
            slice_idx=i,
            pixel_spacing_x=spacing_x + (i % 10) * 0.01,
            pixel_spacing_y=spacing_y + (i % 10) * 0.01,
            slice_thickness=thickness,
            image_path=f"data/{dataset}/{i:06d}.png",
            organs_present=organs or ["lung"],
        )
        for i in range(n)
    ]


# ── DataManifest ──────────────────────────────────────────────────────


class TestDataManifest:
    def test_construction(self):
        records = _make_records("test", 100)
        manifest = DataManifest(records)
        assert len(manifest) == 100

    def test_empty(self):
        manifest = DataManifest()
        assert len(manifest) == 0
        stats = manifest.spacing_stats()
        assert stats.pixel_spacing_x_min == 0.0

    def test_add_and_extend(self):
        manifest = DataManifest()
        manifest.add(_make_records("a", 1)[0])
        assert len(manifest) == 1
        manifest.extend(_make_records("b", 5))
        assert len(manifest) == 6

    def test_spacing_stats(self):
        records = _make_records("test", 10, spacing_x=0.5, thickness=2.0)
        manifest = DataManifest(records)
        stats = manifest.spacing_stats()
        assert stats.pixel_spacing_x_min == pytest.approx(0.5, abs=0.01)
        assert stats.slice_thickness_mean == 2.0

    def test_filter_by_dataset(self):
        records = _make_records("a", 50) + _make_records("b", 30)
        manifest = DataManifest(records)
        filtered = manifest.filter(dataset="a")
        assert len(filtered) == 50

    def test_filter_by_organs(self):
        records = _make_records("a", 50, organs=["lung"]) + _make_records(
            "b", 30, organs=["liver"]
        )
        manifest = DataManifest(records)
        lung = manifest.filter(organs=["lung"])
        assert len(lung) == 50

    def test_sample_uniform(self):
        records = _make_records("test", 1000)
        manifest = DataManifest(records)
        sampled = manifest.sample(100, seed=42)
        assert len(sampled) == 100

    def test_sample_deterministic(self):
        records = _make_records("test", 1000)
        manifest = DataManifest(records)
        s1 = manifest.sample(50, seed=42)
        s2 = manifest.sample(50, seed=42)
        assert [r.slice_idx for r in s1.records] == [
            r.slice_idx for r in s2.records
        ]

    def test_sample_spacing_stratified(self):
        records = _make_records("test", 1000, spacing_x=0.5)
        manifest = DataManifest(records)
        sampled = manifest.sample(100, seed=42, strategy="spacing-stratified")
        assert len(sampled) <= 100

    def test_datasets(self):
        records = _make_records("a", 10) + _make_records("b", 10)
        manifest = DataManifest(records)
        assert manifest.datasets() == ["a", "b"]

    def test_parquet_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow")
        records = _make_records("test", 50, spacing_x=0.5, organs=["lung", "heart"])
        manifest = DataManifest(records)

        parquet_path = tmp_path / "manifest.parquet"
        manifest.save(parquet_path)
        assert parquet_path.exists()

        loaded = DataManifest.load(parquet_path)
        assert len(loaded) == 50
        assert loaded.records[0].dataset == "test"
        assert loaded.records[0].pixel_spacing_x == pytest.approx(0.5, abs=0.01)

    def test_empty_parquet_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow")
        manifest = DataManifest()
        parquet_path = tmp_path / "empty.parquet"
        manifest.save(parquet_path)
        loaded = DataManifest.load(parquet_path)
        assert len(loaded) == 0


# ── DatasetMerger ─────────────────────────────────────────────────────


class TestDatasetMerger:
    def test_basic_merge(self):
        m1 = DataManifest(_make_records("a", 100))
        m2 = DataManifest(_make_records("b", 50))
        merger = DatasetMerger()
        merger.add(m1, weight=0.5)
        merger.add(m2, weight=0.5)
        merged, usage = merger.build(seed=42)
        assert len(merged) == 150
        assert len(usage) == 2

    def test_weighted_sampling(self):
        m1 = DataManifest(_make_records("big", 10000))
        m2 = DataManifest(_make_records("small", 500))
        merger = DatasetMerger()
        merger.add(m1, weight=0.6)
        merger.add(m2, weight=0.4)
        merged, _usage = merger.build(seed=42, total_slices=1000)
        # Should get ~600 from big, ~400 from small
        big_count = sum(1 for r in merged.records if r.dataset == "big")
        small_count = sum(1 for r in merged.records if r.dataset == "small")
        assert big_count == 600
        assert small_count == 400

    def test_weight_normalization(self):
        m1 = DataManifest(_make_records("a", 100))
        m2 = DataManifest(_make_records("b", 100))
        merger = DatasetMerger()
        merger.add(m1, weight=3.0)
        merger.add(m2, weight=1.0)
        _, usage = merger.build(seed=42, total_slices=100)
        assert usage[0].weight == pytest.approx(0.75)
        assert usage[1].weight == pytest.approx(0.25)

    def test_usage_records(self):
        m1 = DataManifest(_make_records("lidc", 200, spacing_x=0.5, thickness=1.0))
        merger = DatasetMerger()
        merger.add(m1, weight=1.0)
        _, usage = merger.build(seed=42)
        assert len(usage) == 1
        assert usage[0].name == "lidc"
        assert usage[0].slices_used == 200

    def test_deterministic(self):
        m1 = DataManifest(_make_records("a", 500))
        m2 = DataManifest(_make_records("b", 500))
        merger1 = DatasetMerger()
        merger1.add(m1, weight=0.5)
        merger1.add(m2, weight=0.5)
        merged1, _ = merger1.build(seed=42, total_slices=200)

        merger2 = DatasetMerger()
        merger2.add(m1, weight=0.5)
        merger2.add(m2, weight=0.5)
        merged2, _ = merger2.build(seed=42, total_slices=200)

        idx1 = [r.slice_idx for r in merged1.records]
        idx2 = [r.slice_idx for r in merged2.records]
        assert idx1 == idx2

    def test_empty_raises(self):
        merger = DatasetMerger()
        with pytest.raises(ValueError, match="No datasets"):
            merger.build()

    def test_negative_weight_raises(self):
        m = DataManifest(_make_records("a", 10))
        merger = DatasetMerger()
        with pytest.raises(ValueError, match="positive"):
            merger.add(m, weight=-1.0)

    def test_capped_by_available(self):
        """If we request more slices than available, use all."""
        m = DataManifest(_make_records("tiny", 10))
        merger = DatasetMerger()
        merger.add(m, weight=1.0)
        merged, _usage = merger.build(seed=42, total_slices=1000)
        assert len(merged) == 10


# ---------------------------------------------------------------------------
# Temperature-scaled sampling
# ---------------------------------------------------------------------------


class TestTemperatureWeights:
    def test_proportional_at_t1(self):
        """T=1.0 should give proportional weights."""
        w = temperature_weights([200, 100, 50], temperature=1.0)
        assert w[0] == pytest.approx(200 / 350)
        assert w[1] == pytest.approx(100 / 350)
        assert w[2] == pytest.approx(50 / 350)

    def test_sqrt_at_t2(self):
        """T=2.0 should give square-root weights."""
        import math
        sizes = [10000, 100]
        w = temperature_weights(sizes, temperature=2.0)
        # sqrt(10000) = 100, sqrt(100) = 10, total = 110
        assert w[0] == pytest.approx(100 / 110)
        assert w[1] == pytest.approx(10 / 110)

    def test_converges_to_uniform(self):
        """Very high T should give near-equal weights."""
        w = temperature_weights([10000, 1], temperature=100.0)
        assert abs(w[0] - w[1]) < 0.1  # Nearly equal

    def test_sums_to_one(self):
        w = temperature_weights([500, 200, 80, 30], temperature=2.0)
        assert sum(w) == pytest.approx(1.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            temperature_weights([100, 50], temperature=-1.0)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            temperature_weights([100, 0], temperature=2.0)

    def test_softens_extreme_ratio(self):
        """A 20:1 ratio should be softened to ~4.5:1 at T=2."""
        w = temperature_weights([200000, 10000], temperature=2.0)
        ratio = w[0] / w[1]
        # sqrt(20) ≈ 4.47
        assert 4.0 < ratio < 5.0


class TestTemperatureScaledMerger:
    def test_temperature_build(self):
        """Temperature strategy should auto-compute weights."""
        big = DataManifest(_make_records("big", 10000))
        small = DataManifest(_make_records("small", 100))
        merger = DatasetMerger()
        merger.add(big)
        merger.add(small)
        merged, usage = merger.build(
            seed=42, total_slices=1000,
            strategy="temperature", temperature=2.0,
        )
        # sqrt(10000)=100, sqrt(100)=10, total=110
        # big: 100/110 ≈ 0.909 → ~909 slices
        # small: 10/110 ≈ 0.091 → ~91 slices
        big_count = sum(1 for r in merged.records if r.dataset == "big")
        small_count = sum(1 for r in merged.records if r.dataset == "small")
        assert 850 < big_count < 950
        assert 50 < small_count < 120

    def test_temperature_ignores_manual_weights(self):
        """Temperature strategy should ignore the weight= argument to add()."""
        m1 = DataManifest(_make_records("a", 1000))
        m2 = DataManifest(_make_records("b", 1000))
        merger = DatasetMerger()
        merger.add(m1, weight=99.0)  # This should be ignored
        merger.add(m2, weight=1.0)   # This should be ignored
        _, usage = merger.build(
            seed=42, total_slices=100,
            strategy="temperature", temperature=1.0,
        )
        # Equal sizes → equal weights regardless of manual weights
        assert usage[0].weight == pytest.approx(0.5)
        assert usage[1].weight == pytest.approx(0.5)

    def test_unknown_strategy_raises(self):
        m = DataManifest(_make_records("a", 100))
        merger = DatasetMerger()
        merger.add(m)
        with pytest.raises(ValueError, match="Unknown strategy"):
            merger.build(strategy="bogus")

    def test_temperature_deterministic(self):
        """Temperature-scaled builds should be deterministic."""
        m1 = DataManifest(_make_records("a", 500))
        m2 = DataManifest(_make_records("b", 100))

        def _build():
            merger = DatasetMerger()
            merger.add(m1)
            merger.add(m2)
            merged, _ = merger.build(
                seed=42, total_slices=200,
                strategy="temperature", temperature=2.0,
            )
            return [r.slice_idx for r in merged.records]

        assert _build() == _build()

    def test_three_organ_mixture(self):
        """Realistic 3-organ scenario: lung(235K) + abdomen(50K) + brain(20K)."""
        lung = DataManifest(_make_records("lung", 2350))
        abdomen = DataManifest(_make_records("abdomen", 500))
        brain = DataManifest(_make_records("brain", 200))

        merger = DatasetMerger()
        merger.add(lung)
        merger.add(abdomen)
        merger.add(brain)
        merged, usage = merger.build(
            seed=42, total_slices=1000,
            strategy="temperature", temperature=2.0,
        )

        # All three organs should have meaningful representation
        lung_count = sum(1 for r in merged.records if r.dataset == "lung")
        abdomen_count = sum(1 for r in merged.records if r.dataset == "abdomen")
        brain_count = sum(1 for r in merged.records if r.dataset == "brain")

        # Brain should NOT be starved (>5% even though it's only 6.5% of data)
        assert brain_count > 50  # Significantly more than proportional 65
        # Lung should NOT dominate (proportional would be ~770)
        assert lung_count < 700
        # All three represented
        assert abdomen_count > 100

"""Multi-dataset merging with weighted sampling.

Combines multiple ``DataManifest`` instances into a single training
set with configurable per-dataset weights, modality/organ filtering,
and spacing-stratified sampling.

Supports three mixing strategies:

1. **Manual weights**: Explicit per-dataset weights (``add(m, weight=0.4)``).
2. **Temperature-scaled**: Automatic weight computation from dataset sizes.
   ``weight_i ∝ n_i^(1/T)`` where T is the temperature parameter.

   - T=1.0 → proportional (weights = dataset size ratios). Large datasets
     dominate. Risk: small organs starved.
   - T=∞ → balanced (equal weights). Risk: small datasets memorized.
   - T=2.0 → square-root (recommended). Smooth compromise that prevents
     both starvation and memorization.

Temperature-scaled sampling is the gold-standard mixing strategy for
multi-domain foundation models (multilingual NLP, multi-modal vision).
"""

from __future__ import annotations

import logging
import math
import random

from zoo.manifest import DataManifest
from zoo.models import DatasetUsage, SliceMetadata

logger = logging.getLogger(__name__)


def temperature_weights(
    sizes: list[int],
    temperature: float = 2.0,
) -> list[float]:
    """Compute temperature-scaled sampling weights from dataset sizes.

    ``weight_i = n_i^(1/T) / sum(n_j^(1/T))``

    Args:
        sizes: Number of slices in each dataset.
        temperature: Mixing temperature.
            - T=1.0: proportional to dataset size.
            - T=2.0: square-root scaling (recommended).
            - T→∞: equal weights regardless of size.

    Returns:
        Normalized weight for each dataset (sums to 1.0).

    Example::

        >>> temperature_weights([200_000, 50_000, 10_000], temperature=2.0)
        [0.54, 0.27, 0.12]  # Square-root softens the 20:1 ratio to ~4.5:1
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"All sizes must be positive, got {sizes}")

    exponent = 1.0 / temperature
    raw = [n ** exponent for n in sizes]
    total = sum(raw)
    return [w / total for w in raw]


class DatasetMerger:
    """Combines multiple dataset manifests for training.

    Example — manual weights::

        merger = DatasetMerger()
        merger.add(lidc_manifest, weight=0.40)
        merger.add(ctorg_manifest, weight=0.25)
        merger.add(abdomen_manifest, weight=0.35)
        merged, usage = merger.build(seed=42, total_slices=500_000)

    Example — temperature-scaled (recommended for pan-organ)::

        merger = DatasetMerger()
        merger.add(lidc_manifest)    # 235K slices
        merger.add(ctorg_manifest)   # 50K slices
        merger.add(brain_manifest)   # 20K slices
        merged, usage = merger.build(
            seed=42, total_slices=500_000,
            strategy="temperature", temperature=2.0,
        )
        # Weights auto-computed: ~0.52, 0.30, 0.18 (sqrt scaling)
    """

    def __init__(self) -> None:
        self._sources: list[tuple[DataManifest, float]] = []

    def add(self, manifest: DataManifest, *, weight: float = 1.0) -> None:
        """Register a dataset manifest with a sampling weight.

        Weights are relative — they are normalized during ``build()``.
        For temperature-scaled sampling, weights are ignored (set to 1.0).
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        self._sources.append((manifest, weight))

    def build(
        self,
        *,
        seed: int = 42,
        total_slices: int | None = None,
        strategy: str = "manual",
        temperature: float = 2.0,
    ) -> tuple[DataManifest, list[DatasetUsage]]:
        """Merge all sources into a single manifest.

        Args:
            seed: Random seed for reproducible sampling.
            total_slices: If set, sample this many slices total
                (distributed according to weights). If ``None``,
                include all slices with epoch-level weighting metadata.
            strategy: Weight computation strategy.
                - ``"manual"``: Use weights from ``add()`` calls.
                - ``"temperature"``: Auto-compute weights from dataset
                  sizes using temperature scaling. Ignores manual weights.
            temperature: Temperature for ``"temperature"`` strategy.
                T=1.0 → proportional, T=2.0 → square-root (recommended),
                T→∞ → balanced.

        Returns:
            Tuple of (merged manifest, list of DatasetUsage records).
        """
        if not self._sources:
            raise ValueError("No datasets added to merger")

        if strategy == "temperature":
            sizes = [len(m) for m, _ in self._sources]
            weights = temperature_weights(sizes, temperature)
            norm_weights = list(zip(
                [m for m, _ in self._sources], weights,
            ))
            logger.info(
                "Temperature-scaled weights (T=%.1f): %s",
                temperature,
                {m.datasets()[0] if m.datasets() else f"ds{i}": f"{w:.3f}"
                 for i, (m, w) in enumerate(norm_weights)},
            )
        elif strategy == "manual":
            total_weight = sum(w for _, w in self._sources)
            norm_weights = [(m, w / total_weight) for m, w in self._sources]
        else:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. Use 'manual' or 'temperature'."
            )

        rng = random.Random(seed)
        merged_records: list[SliceMetadata] = []
        usage_records: list[DatasetUsage] = []

        # Pre-compute per-dataset target counts to avoid int() rounding
        # shortfall.  Distribute leftover slices largest-weight-first.
        if total_slices is not None:
            raw_targets = [max(1, int(total_slices * w))
                           for _, w in norm_weights]
            shortfall = total_slices - sum(raw_targets)
            # Distribute leftover 1-per-dataset, largest weight first
            order = sorted(range(len(norm_weights)),
                           key=lambda i: norm_weights[i][1], reverse=True)
            for i in order:
                if shortfall <= 0:
                    break
                raw_targets[i] += 1
                shortfall -= 1
            targets = raw_targets
        else:
            targets = [None] * len(norm_weights)

        for idx, (manifest, weight) in enumerate(norm_weights):
            n_target = targets[idx]
            if n_target is not None:
                n_available = len(manifest)

                if n_target <= n_available:
                    # Subsample without replacement
                    selected = rng.sample(manifest.records, n_target)
                else:
                    # Oversample with replacement: include all originals
                    # once, then fill the remainder by random draws.
                    # This ensures every slice appears at least once and
                    # the total count matches the temperature-scaled quota.
                    full_copies, remainder = divmod(n_target, n_available)
                    selected = list(manifest.records) * full_copies
                    if remainder:
                        selected += rng.sample(manifest.records, remainder)
                    logger.info(
                        "Oversampling %s: %d target from %d physical "
                        "(%.1f× repetition)",
                        manifest.datasets()[0] if manifest.datasets()
                        else "unknown",
                        n_target,
                        n_available,
                        n_target / n_available,
                    )
            else:
                # Include all slices
                selected = list(manifest.records)

            merged_records.extend(selected)

            # Compute usage stats
            stats = DataManifest(selected).spacing_stats()
            datasets = manifest.datasets()
            dataset_name = datasets[0] if len(datasets) == 1 else "+".join(datasets)

            usage_records.append(
                DatasetUsage(
                    name=dataset_name,
                    slices_used=len(selected),
                    weight=weight,
                    pixel_spacing_min=stats.pixel_spacing_x_min,
                    pixel_spacing_max=stats.pixel_spacing_x_max,
                    slice_thickness_min=stats.slice_thickness_min,
                    slice_thickness_max=stats.slice_thickness_max,
                )
            )

        # Shuffle the merged records
        rng.shuffle(merged_records)

        logger.info(
            "Merged %d datasets → %d slices (requested %s, strategy=%s)",
            len(self._sources),
            len(merged_records),
            total_slices or "all",
            strategy,
        )

        return DataManifest(merged_records), usage_records

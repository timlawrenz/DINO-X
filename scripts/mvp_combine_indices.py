#!/usr/bin/env python3
"""Combine multiple dataset index CSVs into one training-ready CSV.

Merges index CSVs from multiple processed datasets (e.g., LIDC-IDRI + Pancreas-CT)
into a single CSV with a ``dataset`` column. Optionally subsamples by series count.

The output CSV is directly consumable by ``phase5_big_run.py`` with ``--scale-aware``.

Usage:
  python mvp_combine_indices.py \
    --inputs lidc-idri:/path/to/lidc/index_with_spacing.csv \
    --inputs pancreas-ct:/path/to/pancreas/index.csv \
    --out /path/to/combined/index.csv \
    --max-series-per-dataset 100
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def load_index(
    path: Path,
    dataset_name: str,
    max_series: int = 0,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Load an index CSV and tag rows with the dataset name.

    If max_series > 0, randomly sample that many unique series_dir values.
    """
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"WARNING: {path} is empty", file=sys.stderr)
        return []

    # Tag each row with dataset name
    for row in rows:
        row["dataset"] = dataset_name

    # Subsample by series
    if max_series > 0:
        all_series = sorted({r["series_dir"] for r in rows})
        if len(all_series) > max_series:
            rng = random.Random(seed)
            keep = set(rng.sample(all_series, max_series))
            rows = [r for r in rows if r["series_dir"] in keep]
            print(f"  {dataset_name}: subsampled {max_series}/{len(all_series)} series "
                  f"-> {len(rows)} slices")
        else:
            print(f"  {dataset_name}: all {len(all_series)} series ({len(rows)} slices)")
    else:
        all_series = {r["series_dir"] for r in rows}
        print(f"  {dataset_name}: {len(all_series)} series ({len(rows)} slices)")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Combine dataset index CSVs")
    ap.add_argument(
        "--inputs", action="append", required=True,
        help="dataset_name:/path/to/index.csv (repeat for each dataset)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output combined CSV path")
    ap.add_argument("--max-series-per-dataset", type=int, default=0,
                    help="Max series to sample per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_rows: list[dict[str, str]] = []
    output_columns = [
        "png_path", "series_dir", "slice_index", "encoding",
        "spacing_x", "spacing_y", "spacing_z", "dataset",
    ]

    for input_spec in args.inputs:
        if ":" not in input_spec:
            print(f"ERROR: --inputs must be name:/path format, got: {input_spec}",
                  file=sys.stderr)
            return 1
        name, path_str = input_spec.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"ERROR: index not found: {path}", file=sys.stderr)
            return 1

        rows = load_index(path, name, max_series=args.max_series_per_dataset, seed=args.seed)

        # Ensure spacing columns exist
        for row in rows:
            row.setdefault("spacing_x", "1.000000")
            row.setdefault("spacing_y", "1.000000")
            row.setdefault("spacing_z", "1.000000")

        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no rows collected", file=sys.stderr)
        return 1

    # Write combined CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    datasets = {}
    for r in all_rows:
        ds = r["dataset"]
        datasets.setdefault(ds, {"slices": 0, "series": set()})
        datasets[ds]["slices"] += 1
        datasets[ds]["series"].add(r["series_dir"])

    print(f"\nCombined index: {len(all_rows)} total slices")
    for ds, info in sorted(datasets.items()):
        print(f"  {ds}: {info['slices']} slices, {len(info['series'])} series")
    print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 4: Create a deterministic train/validation split manifest.

This operates on Phase 2's `data/processed/_index/index.csv` and produces a small
JSON manifest intended to be referenced by training/monitoring entrypoints.

Design notes:
- We split at `series_dir` granularity to avoid leakage across adjacent slices.
- The manifest stores split membership as `series_dir` strings (not per-slice
  lists) to keep it compact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: data/processed/_splits/val10_seed<seed>.json",
    )
    args = ap.parse_args()

    if not args.index_csv.exists():
        raise SystemExit(f"index_csv not found: {args.index_csv}")

    if not (0.0 < args.val_frac < 1.0):
        raise SystemExit("--val-frac must be in (0,1)")

    series_counts: dict[str, int] = {}
    total_rows = 0

    with args.index_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if "series_dir" not in fields:
            raise SystemExit(f"index_csv missing series_dir column: {args.index_csv}")
        for row in r:
            total_rows += 1
            sd = row.get("series_dir", "")
            series_counts[sd] = series_counts.get(sd, 0) + 1

    if not series_counts:
        raise SystemExit(f"No rows/series found in index_csv: {args.index_csv}")

    import random

    groups = list(series_counts.keys())
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_val = max(1, int(math.ceil(args.val_frac * n_groups)))
    val_groups = sorted(groups[:n_val])
    train_groups = sorted(groups[n_val:])

    val_images = sum(series_counts[g] for g in val_groups)
    train_images = sum(series_counts[g] for g in train_groups)

    out = args.out
    if out is None:
        out = Path("data/processed/_splits") / f"val10_seed{args.seed}.json"

    payload = {
        "kind": "dataset_split",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_csv": str(args.index_csv),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "group_key": "series_dir",
        "counts": {
            "groups_total": n_groups,
            "groups_train": len(train_groups),
            "groups_val": len(val_groups),
            "images_total": total_rows,
            "images_train": train_images,
            "images_val": val_images,
        },
        "train": {"series_dir": train_groups},
        "val": {"series_dir": val_groups},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("ok=true")
    print(f"out={out}")
    print(f"groups_total={n_groups}")
    print(f"images_total={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

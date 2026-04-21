#!/usr/bin/env python3
"""Extract DICOM spacing metadata for an already-processed dataset.

Scans raw DICOM series directories, reads one DICOM per series
(stop_before_pixels for speed), and extracts:
  - PixelSpacing[0] -> spacing_x (mm)
  - PixelSpacing[1] -> spacing_y (mm)
  - SliceThickness  -> spacing_z (mm)

Outputs a per-series spacing CSV, then merges it with the existing
index.csv to produce index_with_spacing.csv.

Usage:
  python extract_dicom_spacing.py \
    --dicom-root /mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw \
    --index-csv /mnt/nas-ai-models/training-data/dino-x/lidc-idri/processed-hu16/_index/index.csv \
    --out-dir /mnt/nas-ai-models/training-data/dino-x/lidc-idri/processed-hu16/_index
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. pip install {mod}")


try:
    import pydicom  # type: ignore
except ImportError:
    _need("pydicom")


def extract_series_spacing(series_dir: Path) -> dict[str, float] | None:
    """Read DICOM files from a series dir to extract spacing metadata.

    For z-spacing, computes the median delta between adjacent
    ImagePositionPatient z-coordinates (more reliable than SliceThickness).
    Falls back to SpacingBetweenSlices, then SliceThickness.

    Returns dict with spacing_x, spacing_y, spacing_z or None on failure.
    """
    dicom_files: list[Path] = []
    for root, _, fnames in os.walk(series_dir):
        for fname in sorted(fnames):
            fp = Path(root) / fname
            if not fp.is_file() or fp.suffix in (".complete", ".txt"):
                continue
            if fp.name == "LICENSE":
                continue
            dicom_files.append(fp)

    if not dicom_files:
        return None

    spacing_x: float | None = None
    spacing_y: float | None = None
    slice_thickness: float | None = None
    spacing_between: float | None = None
    z_positions: list[float] = []

    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue

        if not hasattr(ds, "Rows"):
            continue

        # Extract in-plane spacing (only need from first valid DICOM)
        if spacing_x is None:
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None and len(ps) >= 2:
                spacing_x = float(ps[0])
                spacing_y = float(ps[1])

            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                slice_thickness = float(st)

            sbs = getattr(ds, "SpacingBetweenSlices", None)
            if sbs is not None:
                spacing_between = float(sbs)

        # Collect z-positions for computing actual inter-slice spacing
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z_positions.append(float(ipp[2]))

    if spacing_x is None:
        return None

    # Compute z-spacing: prefer IPP deltas > SpacingBetweenSlices > SliceThickness
    spacing_z: float
    if len(z_positions) >= 2:
        z_sorted = sorted(z_positions)
        deltas = [abs(z_sorted[i + 1] - z_sorted[i])
                  for i in range(len(z_sorted) - 1)]
        # Use median to handle any duplicates or outliers
        deltas.sort()
        spacing_z = deltas[len(deltas) // 2] if deltas else 1.0
        if spacing_z == 0.0:
            spacing_z = spacing_between or slice_thickness or 1.0
    elif spacing_between is not None:
        spacing_z = spacing_between
    elif slice_thickness is not None:
        spacing_z = slice_thickness
    else:
        spacing_z = 1.0

    return {
        "spacing_x": spacing_x,
        "spacing_y": spacing_y or spacing_x,
        "spacing_z": spacing_z,
    }


def iter_series_dirs(dicom_root: Path):
    """Yield directories containing DICOM files."""
    for root, dirnames, fnames in os.walk(dicom_root):
        has_files = any(
            not f.endswith((".complete", ".txt", "LICENSE"))
            for f in fnames
        )
        if has_files:
            yield Path(root)
            dirnames[:] = []


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract DICOM spacing metadata")
    ap.add_argument("--dicom-root", type=Path, required=True,
                    help="Root dir containing raw DICOM series subdirectories")
    ap.add_argument("--index-csv", type=Path, default=None,
                    help="Existing index.csv to merge spacing into")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: same as index-csv parent)")
    args = ap.parse_args()

    dicom_root = args.dicom_root
    if not dicom_root.exists():
        print(f"ERROR: dicom-root not found: {dicom_root}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (args.index_csv.parent if args.index_csv else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Extract spacing per series ---
    spacing_csv = out_dir / "spacing_by_series.csv"
    print(f"Scanning DICOM series in {dicom_root} ...")

    series_spacing: dict[str, dict[str, float]] = {}
    n_ok = 0
    n_fail = 0

    for series_dir in sorted(iter_series_dirs(dicom_root)):
        if series_dir.name.startswith("_"):
            continue

        spacing = extract_series_spacing(series_dir)
        if spacing is None:
            n_fail += 1
            print(f"  SKIP {series_dir.name} (no readable DICOM)")
            continue

        series_spacing[str(series_dir)] = spacing
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  ... extracted {n_ok} series")

    print(f"Extracted spacing from {n_ok} series ({n_fail} failed)")

    with spacing_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_dir", "spacing_x", "spacing_y", "spacing_z"])
        for sd, sp in sorted(series_spacing.items()):
            w.writerow([sd, f"{sp['spacing_x']:.6f}", f"{sp['spacing_y']:.6f}", f"{sp['spacing_z']:.6f}"])

    print(f"Wrote {spacing_csv}")

    # --- Phase 2: Merge with existing index CSV ---
    if args.index_csv is None:
        print("No --index-csv provided; skipping merge.")
        return 0

    if not args.index_csv.exists():
        print(f"ERROR: index-csv not found: {args.index_csv}", file=sys.stderr)
        return 1

    # Build a lookup: series_dir string -> spacing.
    # Normalize paths for robust matching.
    uid_to_spacing: dict[str, dict[str, float]] = {}
    basename_counts: dict[str, int] = {}
    for sd_path, sp in series_spacing.items():
        normalized = str(Path(sd_path).resolve())
        uid_to_spacing[normalized] = sp
        uid_to_spacing[sd_path] = sp
        bn = Path(sd_path).name
        basename_counts[bn] = basename_counts.get(bn, 0) + 1
        if basename_counts[bn] == 1:
            uid_to_spacing[bn] = sp
        else:
            # Basename collision — remove to prevent wrong matches
            uid_to_spacing.pop(bn, None)

    merged_csv = out_dir / "index_with_spacing.csv"
    n_matched = 0
    n_unmatched = 0
    n_total = 0
    unmatched_series: set[str] = set()

    with args.index_csv.open("r", newline="") as fin, \
         merged_csv.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["png_path", "series_dir", "slice_index", "encoding",
                         "spacing_x", "spacing_y", "spacing_z"])

        for row in reader:
            n_total += 1
            sd = row["series_dir"]
            # Try matching by exact path, resolved path, then basename
            sp = (uid_to_spacing.get(sd)
                  or uid_to_spacing.get(str(Path(sd).resolve()))
                  or uid_to_spacing.get(Path(sd).name))

            if sp is not None:
                n_matched += 1
                writer.writerow([
                    row["png_path"], row["series_dir"],
                    row["slice_index"], row["encoding"],
                    f"{sp['spacing_x']:.6f}",
                    f"{sp['spacing_y']:.6f}",
                    f"{sp['spacing_z']:.6f}",
                ])
            else:
                n_unmatched += 1
                unmatched_series.add(sd)
                writer.writerow([
                    row["png_path"], row["series_dir"],
                    row["slice_index"], row["encoding"],
                    "1.000000", "1.000000", "1.000000",
                ])

    # Fail loudly if too many rows are unmatched
    unmatched_pct = (n_unmatched / n_total * 100) if n_total > 0 else 0
    print(f"Merged {n_total} rows: {n_matched} matched, "
          f"{n_unmatched} unmatched ({unmatched_pct:.1f}%)")

    if unmatched_series:
        print(f"\nWARNING: {len(unmatched_series)} series had no spacing match:",
              file=sys.stderr)
        for s in sorted(unmatched_series)[:10]:
            print(f"  {s}", file=sys.stderr)
        if len(unmatched_series) > 10:
            print(f"  ... and {len(unmatched_series) - 10} more",
                  file=sys.stderr)

    if unmatched_pct > 10:
        print(f"\nERROR: {unmatched_pct:.1f}% unmatched rows — "
              "spacing data will be unreliable for scale-aware training.",
              file=sys.stderr)
        return 1

    print(f"Wrote {merged_csv}")

    # Summary stats
    if series_spacing:
        xs = [s["spacing_x"] for s in series_spacing.values()]
        ys = [s["spacing_y"] for s in series_spacing.values()]
        zs = [s["spacing_z"] for s in series_spacing.values()]
        print(f"\nSpacing statistics ({n_ok} series):")
        print(f"  pixel_spacing_x: min={min(xs):.4f} max={max(xs):.4f} mean={sum(xs)/len(xs):.4f}")
        print(f"  pixel_spacing_y: min={min(ys):.4f} max={max(ys):.4f} mean={sum(ys)/len(ys):.4f}")
        print(f"  slice_thickness:  min={min(zs):.4f} max={max(zs):.4f} mean={sum(zs)/len(zs):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

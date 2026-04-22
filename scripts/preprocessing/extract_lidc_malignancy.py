#!/usr/bin/env python3
"""Extract LIDC-IDRI nodule malignancy labels for LoRA fine-tuning.

Uses pylidc to parse the bundled LIDC-IDRI annotation database, maps
annotated nodules to our preprocessed PNG slices, and creates
nodule-centered crops + train/val/test CSVs with patient-level splits.

Two modes:
  --crop-dir: Extract nodule-centered crops from HU16 PNGs (recommended).
              Each crop is padded to at least --min-crop-size pixels.
  Without --crop-dir: Output whole-slice paths (less effective for LoRA).

Output CSV format (compatible with scripts/finetune_lora.py):
    image_path,label,spacing_x,spacing_y,spacing_z,patient_id,avg_malignancy,n_raters,rater_agreement

Usage::

    python scripts/preprocessing/extract_lidc_malignancy.py \\
      --index-csv data/processed/_index/index_with_spacing.csv \\
      --output-dir data/lidc-idri/labels \\
      --threshold 3.0 --min-raters 2 \\
      --train-ratio 0.70 --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Monkey-patch for numpy compat with pylidc
np.int = np.int64  # type: ignore[attr-defined]

import pylidc as pl  # noqa: E402


@dataclass
class NoduleRecord:
    """One labeled slice for a nodule centroid."""

    image_path: str
    label: int
    spacing_x: float
    spacing_y: float
    spacing_z: float
    patient_id: str
    avg_malignancy: float
    n_raters: int
    rater_agreement: float  # std of malignancy ratings
    # Bounding box in pixel coords (for reference/debugging)
    bbox_imin: int = 0
    bbox_imax: int = 0
    bbox_jmin: int = 0
    bbox_jmax: int = 0


def load_index(index_csv: Path) -> dict[str, dict[int, tuple[str, float, float, float]]]:
    """Load preprocessed index CSV.

    Returns: {series_dir: {slice_index: (png_path, spacing_x, y, z)}}
    """
    series_map: dict[str, dict[int, tuple[str, float, float, float]]] = defaultdict(dict)

    with open(index_csv) as f:
        reader = csv.DictReader(f)
        has_spacing = all(
            c in (reader.fieldnames or [])
            for c in ("spacing_x", "spacing_y", "spacing_z")
        )

        for row in reader:
            sd = row["series_dir"]
            si = int(row["slice_index"])
            pp = row["png_path"]
            sx = float(row["spacing_x"]) if has_spacing else 1.0
            sy = float(row["spacing_y"]) if has_spacing else 1.0
            sz = float(row["spacing_z"]) if has_spacing else 1.0
            series_map[sd][si] = (pp, sx, sy, sz)

    return dict(series_map)


def _save_nodule_crop(
    src_path: Path,
    crop_dir: Path,
    nodule_id: int,
    bbox_imin: int,
    bbox_imax: int,
    bbox_jmin: int,
    bbox_jmax: int,
    min_crop_size: int = 64,
) -> Path | None:
    """Crop a nodule region from a HU16 PNG and save it."""
    try:
        img = Image.open(src_path)
    except Exception:
        return None

    w, h = img.size  # PIL: (width, height) = (cols, rows)

    # Calculate padded crop (ensure minimum size, centered on nodule)
    nod_h = bbox_imax - bbox_imin
    nod_w = bbox_jmax - bbox_jmin
    crop_h = max(nod_h * 2, min_crop_size)  # 2× nodule size or minimum
    crop_w = max(nod_w * 2, min_crop_size)

    center_i = (bbox_imin + bbox_imax) // 2
    center_j = (bbox_jmin + bbox_jmax) // 2

    # Clamp to image bounds
    i0 = max(0, center_i - crop_h // 2)
    i1 = min(h, i0 + crop_h)
    i0 = max(0, i1 - crop_h)

    j0 = max(0, center_j - crop_w // 2)
    j1 = min(w, j0 + crop_w)
    j0 = max(0, j1 - crop_w)

    # PIL crop uses (left, upper, right, lower) = (j0, i0, j1, i1)
    cropped = img.crop((j0, i0, j1, i1))

    out_path = crop_dir / f"nodule_{nodule_id:05d}.png"
    cropped.save(out_path)
    return out_path


def extract_nodule_records(
    series_map: dict[str, dict[int, tuple[str, float, float, float]]],
    threshold: float = 3.0,
    min_raters: int = 2,
    crop_dir: Path | None = None,
    min_crop_size: int = 64,
    data_root: Path = Path("."),
) -> list[NoduleRecord]:
    """Extract labeled nodule records from pylidc annotations.

    If crop_dir is provided, extracts nodule-centered crops from
    the HU16 PNGs and saves them. Otherwise, returns whole-slice paths.
    """

    scans = pl.query(pl.Scan).all()
    logger.info("Processing %d LIDC scans from pylidc database", len(scans))

    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving nodule crops to %s (min size %dpx)", crop_dir, min_crop_size)

    records: list[NoduleRecord] = []
    skipped_no_series = 0
    skipped_count_mismatch = 0
    skipped_few_raters = 0
    skipped_bad_centroid = 0
    nodule_id = 0

    for scan in scans:
        uid = scan.series_instance_uid
        series_dir = f"data_raw_{uid}"

        if series_dir not in series_map:
            skipped_no_series += 1
            continue

        our_slices = series_map[series_dir]
        pylidc_count = len(scan.slice_zvals)

        if len(our_slices) != pylidc_count:
            skipped_count_mismatch += 1
            logger.debug(
                "Slice count mismatch for %s: ours=%d pylidc=%d",
                scan.patient_id, len(our_slices), pylidc_count,
            )
            continue

        clusters = scan.cluster_annotations()

        for cluster in clusters:
            if len(cluster) < min_raters:
                skipped_few_raters += 1
                continue

            malignancies = [a.malignancy for a in cluster]
            avg_mal = float(np.mean(malignancies))
            std_mal = float(np.std(malignancies))
            n_raters = len(cluster)

            # Get centroid and bounding box from contours
            try:
                all_contours = np.vstack([a.contours_matrix for a in cluster])
                centroid = all_contours.mean(axis=0)
                centroid_k = int(round(centroid[2]))

                # Bounding box on centroid slice (union of all annotators)
                slice_mask = all_contours[:, 2] == centroid_k
                if slice_mask.sum() == 0:
                    slice_contours = all_contours
                else:
                    slice_contours = all_contours[slice_mask]

                bbox_imin = int(slice_contours[:, 0].min())
                bbox_imax = int(slice_contours[:, 0].max())
                bbox_jmin = int(slice_contours[:, 1].min())
                bbox_jmax = int(slice_contours[:, 1].max())
            except Exception:
                skipped_bad_centroid += 1
                continue

            centroid_k = max(0, min(centroid_k, pylidc_count - 1))

            if centroid_k not in our_slices:
                skipped_bad_centroid += 1
                continue

            png_path, sx, sy, sz = our_slices[centroid_k]
            label = 1 if avg_mal >= threshold else 0

            # Extract crop if requested
            if crop_dir is not None:
                crop_path = _save_nodule_crop(
                    data_root / png_path, crop_dir, nodule_id,
                    bbox_imin, bbox_imax, bbox_jmin, bbox_jmax,
                    min_crop_size,
                )
                if crop_path is None:
                    skipped_bad_centroid += 1
                    continue
                final_path = str(crop_path)
            else:
                final_path = png_path

            records.append(NoduleRecord(
                image_path=final_path,
                label=label,
                spacing_x=sx,
                spacing_y=sy,
                spacing_z=sz,
                patient_id=scan.patient_id,
                avg_malignancy=round(avg_mal, 2),
                n_raters=n_raters,
                rater_agreement=round(std_mal, 2),
                bbox_imin=bbox_imin,
                bbox_imax=bbox_imax,
                bbox_jmin=bbox_jmin,
                bbox_jmax=bbox_jmax,
            ))
            nodule_id += 1

    logger.info("Extraction complete:")
    logger.info("  Total records: %d", len(records))
    logger.info("  Skipped (no series in index): %d", skipped_no_series)
    logger.info("  Skipped (slice count mismatch): %d", skipped_count_mismatch)
    logger.info("  Skipped (too few raters): %d", skipped_few_raters)
    logger.info("  Skipped (bad centroid): %d", skipped_bad_centroid)

    # Label distribution
    pos = sum(1 for r in records if r.label == 1)
    neg = sum(1 for r in records if r.label == 0)
    logger.info("  Positive (malignancy >= %.1f): %d (%.1f%%)",
                threshold, pos, 100 * pos / max(len(records), 1))
    logger.info("  Negative (malignancy <  %.1f): %d (%.1f%%)",
                threshold, neg, 100 * neg / max(len(records), 1))

    return records


def patient_stratified_split(
    records: list[NoduleRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[NoduleRecord], list[NoduleRecord], list[NoduleRecord]]:
    """Split records into train/val/test by patient, stratified by label.

    Patients (not nodules) are the splitting unit to prevent data leakage.
    Stratification ensures balanced label distribution across splits.
    """
    rng = np.random.RandomState(seed)

    # Group records by patient and determine majority label
    patient_records: dict[str, list[NoduleRecord]] = defaultdict(list)
    for r in records:
        patient_records[r.patient_id].append(r)

    # Patient majority label (for stratification)
    patient_label: dict[str, int] = {}
    for pid, recs in patient_records.items():
        labels = [r.label for r in recs]
        patient_label[pid] = 1 if sum(labels) > len(labels) / 2 else 0

    # Separate patients by majority label
    pos_patients = [p for p, l in patient_label.items() if l == 1]
    neg_patients = [p for p, l in patient_label.items() if l == 0]

    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def _split_list(patients: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(patients)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            patients[:n_train],
            patients[n_train:n_train + n_val],
            patients[n_train + n_val:],
        )

    pos_train, pos_val, pos_test = _split_list(pos_patients)
    neg_train, neg_val, neg_test = _split_list(neg_patients)

    train_pids = set(pos_train + neg_train)
    val_pids = set(pos_val + neg_val)
    test_pids = set(pos_test + neg_test)

    train = [r for r in records if r.patient_id in train_pids]
    val = [r for r in records if r.patient_id in val_pids]
    test = [r for r in records if r.patient_id in test_pids]

    logger.info("Patient-level split (seed=%d):", seed)
    logger.info("  Train: %d patients, %d nodules (pos=%d neg=%d)",
                len(train_pids), len(train),
                sum(1 for r in train if r.label == 1),
                sum(1 for r in train if r.label == 0))
    logger.info("  Val:   %d patients, %d nodules (pos=%d neg=%d)",
                len(val_pids), len(val),
                sum(1 for r in val if r.label == 1),
                sum(1 for r in val if r.label == 0))
    logger.info("  Test:  %d patients, %d nodules (pos=%d neg=%d)",
                len(test_pids), len(test),
                sum(1 for r in test if r.label == 1),
                sum(1 for r in test if r.label == 0))

    return train, val, test


def write_csv(records: list[NoduleRecord], path: Path) -> None:
    """Write nodule records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "label", "spacing_x", "spacing_y", "spacing_z",
        "patient_id", "avg_malignancy", "n_raters", "rater_agreement",
        "bbox_imin", "bbox_imax", "bbox_jmin", "bbox_jmax",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "image_path": r.image_path,
                "label": r.label,
                "spacing_x": r.spacing_x,
                "spacing_y": r.spacing_y,
                "spacing_z": r.spacing_z,
                "patient_id": r.patient_id,
                "avg_malignancy": r.avg_malignancy,
                "n_raters": r.n_raters,
                "rater_agreement": r.rater_agreement,
                "bbox_imin": r.bbox_imin,
                "bbox_imax": r.bbox_imax,
                "bbox_jmin": r.bbox_jmin,
                "bbox_jmax": r.bbox_jmax,
            })

    logger.info("Wrote %d records to %s", len(records), path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract LIDC-IDRI nodule malignancy labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--index-csv", type=Path,
        default=Path("data/processed/_index/index_with_spacing.csv"),
        help="Path to preprocessed index CSV with spacing",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/lidc-idri/labels"),
        help="Output directory for train/val/test CSVs",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="Malignancy threshold for binary label (>= threshold → positive)",
    )
    parser.add_argument(
        "--min-raters", type=int, default=2,
        help="Minimum number of radiologist annotations per nodule",
    )
    parser.add_argument(
        "--crop-dir", type=Path, default=None,
        help="If set, extract nodule-centered crops to this directory",
    )
    parser.add_argument(
        "--min-crop-size", type=int, default=64,
        help="Minimum crop size in pixels (nodule bbox is padded to this)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("."),
        help="Root directory for resolving relative image paths",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    logger.info("Loading index from %s", args.index_csv)
    series_map = load_index(args.index_csv)
    logger.info("Loaded %d series from index", len(series_map))

    records = extract_nodule_records(
        series_map,
        threshold=args.threshold,
        min_raters=args.min_raters,
        crop_dir=args.crop_dir,
        min_crop_size=args.min_crop_size,
        data_root=args.data_root,
    )

    if not records:
        logger.error("No records extracted! Check index CSV and pylidc database.")
        sys.exit(1)

    train, val, test = patient_stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_csv(train, args.output_dir / "malignancy_train.csv")
    write_csv(val, args.output_dir / "malignancy_val.csv")
    write_csv(test, args.output_dir / "malignancy_test.csv")

    # Also write the full dataset for reference
    write_csv(records, args.output_dir / "malignancy_all.csv")

    logger.info("Done! Label extraction complete.")


if __name__ == "__main__":
    main()

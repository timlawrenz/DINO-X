#!/usr/bin/env python3
"""Extract binary labels from MSD NIfTI segmentation masks for LoRA fine-tuning.

MSD datasets (colon, hepatic-vessel) have 3D NIfTI segmentation masks where
label=1 indicates tumor/lesion presence. This script:
1. Loads the 3D segmentation mask for each patient
2. Finds which slices contain the label (positive) vs. don't (negative)
3. Maps to preprocessed PNG slice paths from the index CSV
4. Creates train/val/test CSVs with patient-level splits

Output CSV format (compatible with scripts/finetune_lora.py):
    image_path,label,spacing_x,spacing_y,spacing_z,patient_id,tumor_pixels,slice_has_tumor

Usage::

    python scripts/preprocessing/extract_msd_labels.py \
      --dataset msd-colon \
      --raw-dir /mnt/nas-ai-models/training-data/dino-x/raw/msd-colon/Task10_Colon \
      --index-csv data/processed/msd-colon/index.csv \
      --output-dir data/msd-colon/labels \
      --train-ratio 0.70 --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MSDRecord:
    """One labeled slice for an MSD dataset."""

    image_path: str
    label: int  # 1 if tumor present in slice, 0 otherwise
    spacing_x: float
    spacing_y: float
    spacing_z: float
    patient_id: str
    tumor_pixels: int  # Number of tumor pixels in this slice
    slice_has_tumor: bool  # Same as label, kept for clarity


def load_index(index_csv: Path) -> dict[str, dict[int, tuple[str, float, float, float]]]:
    """Load preprocessed index CSV.

    Returns: {series_dir: {slice_index: (png_path, spacing_x, y, z)}}
    """
    series_map: dict[str, dict[int, tuple[str, float, float, float]]] = defaultdict(dict)

    with open(index_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sd = row["series_dir"]
            si = int(row["slice_index"])
            pp = row["png_path"]
            sx = float(row["spacing_x"])
            sy = float(row["spacing_y"])
            sz = float(row["spacing_z"])
            series_map[sd][si] = (pp, sx, sy, sz)

    return dict(series_map)


def load_nifti_labels(label_path: Path) -> np.ndarray:
    """Load a 3D NIfTI segmentation mask and return as numpy array.

    Returns: 3D array with shape (H, W, D) where D is the number of slices.
    """
    nii = nib.load(str(label_path))
    data = nii.get_fdata()
    # Ensure integer type (some NIfTI files store as float)
    return data.astype(np.uint8)


def find_tumor_slices(mask_3d: np.ndarray, min_tumor_pixels: int = 10) -> dict[int, int]:
    """Find which slices contain tumor and count pixels.

    Args:
        mask_3d: 3D segmentation mask (H, W, D)
        min_tumor_pixels: Minimum pixels to consider a slice positive

    Returns: {slice_index: tumor_pixel_count}
    """
    tumor_slices: dict[int, int] = {}

    # Sum over H and W dimensions to get per-slice tumor pixel counts
    slice_sums = mask_3d.sum(axis=(0, 1))

    for slice_idx, count in enumerate(slice_sums):
        if count >= min_tumor_pixels:
            tumor_slices[slice_idx] = int(count)

    return tumor_slices


def extract_msd_records(
    series_map: dict[str, dict[int, tuple[str, float, float, float]]],
    labels_dir: Path,
    dataset_prefix: str,
    min_tumor_pixels: int = 10,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> list[MSDRecord]:
    """Extract labeled records from MSD NIfTI segmentation masks.

    Args:
        series_map: Preprocessed index {series_dir: {slice_index: (png_path, sx, sy, sz)}}
        labels_dir: Directory containing NIfTI label files
        dataset_prefix: Prefix for patient IDs (e.g., "colon" or "hepaticvessel")
        min_tumor_pixels: Minimum pixels to consider a slice positive
        negative_ratio: Ratio of negative to positive slices to include
        seed: Random seed for negative sampling

    Returns: List of MSDRecord objects
    """
    rng = np.random.RandomState(seed)

    # Find all label files
    label_files = sorted(labels_dir.glob(f"{dataset_prefix}_*.nii.gz"))
    # Filter out macOS metadata files (._*)
    label_files = [f for f in label_files if not f.name.startswith("._")]

    logger.info("Found %d label files in %s", len(label_files), labels_dir)

    records: list[MSDRecord] = []
    skipped_no_series = 0
    skipped_no_labels = 0
    skipped_slice_mismatch = 0

    for label_path in label_files:
        # Extract patient ID from filename (e.g., "colon_219.nii.gz" -> "219")
        stem = label_path.stem.replace(".nii", "")  # Remove .nii.gz
        patient_num = stem.split("_")[-1]
        patient_id = f"{dataset_prefix}_{patient_num}"

        # Find matching series in preprocessed index
        # Series dir format: "msd-colon_colon_001" or "msd-hepatic-vessel_hepaticvessel_001"
        series_dir = None
        for sd in series_map.keys():
            if patient_num in sd:
                series_dir = sd
                break

        if series_dir is None:
            skipped_no_series += 1
            continue

        our_slices = series_map[series_dir]

        # Load 3D mask
        try:
            mask_3d = load_nifti_labels(label_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", label_path, e)
            skipped_no_labels += 1
            continue

        # Find tumor slices
        tumor_slices = find_tumor_slices(mask_3d, min_tumor_pixels)

        if not tumor_slices:
            skipped_no_labels += 1
            continue

        # Check slice count match
        n_slices_3d = mask_3d.shape[2]
        n_slices_index = len(our_slices)

        if n_slices_3d != n_slices_index:
            logger.warning(
                "Slice count mismatch for %s: mask=%d index=%d",
                patient_id, n_slices_3d, n_slices_index
            )
            skipped_slice_mismatch += 1
            continue

        # Extract positive slices
        positive_records = []
        for slice_idx, tumor_pixels in tumor_slices.items():
            if slice_idx not in our_slices:
                continue

            png_path, sx, sy, sz = our_slices[slice_idx]
            positive_records.append(MSDRecord(
                image_path=png_path,
                label=1,
                spacing_x=sx,
                spacing_y=sy,
                spacing_z=sz,
                patient_id=patient_id,
                tumor_pixels=tumor_pixels,
                slice_has_tumor=True,
            ))

        # Sample negative slices (same patient, no tumor)
        all_slice_indices = set(our_slices.keys())
        negative_indices = list(all_slice_indices - set(tumor_slices.keys()))

        n_negatives = int(len(positive_records) * negative_ratio)
        if n_negatives > 0 and negative_indices:
            sampled_negatives = rng.choice(
                negative_indices,
                size=min(n_negatives, len(negative_indices)),
                replace=False
            )

            for slice_idx in sampled_negatives:
                png_path, sx, sy, sz = our_slices[slice_idx]
                records.append(MSDRecord(
                    image_path=png_path,
                    label=0,
                    spacing_x=sx,
                    spacing_y=sy,
                    spacing_z=sz,
                    patient_id=patient_id,
                    tumor_pixels=0,
                    slice_has_tumor=False,
                ))

        records.extend(positive_records)

    logger.info("Extraction complete:")
    logger.info("  Total records: %d", len(records))
    logger.info("  Skipped (no series in index): %d", skipped_no_series)
    logger.info("  Skipped (no tumor in mask): %d", skipped_no_labels)
    logger.info("  Skipped (slice count mismatch): %d", skipped_slice_mismatch)

    # Label distribution
    pos = sum(1 for r in records if r.label == 1)
    neg = sum(1 for r in records if r.label == 0)
    logger.info("  Positive (tumor): %d (%.1f%%)", pos, 100 * pos / max(len(records), 1))
    logger.info("  Negative (no tumor): %d (%.1f%%)", neg, 100 * neg / max(len(records), 1))

    return records


def patient_stratified_split(
    records: list[MSDRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[MSDRecord], list[MSDRecord], list[MSDRecord]]:
    """Split records into train/val/test by patient, stratified by label.

    Patients (not slices) are the splitting unit to prevent data leakage.
    Stratification ensures balanced label distribution across splits.
    """
    rng = np.random.RandomState(seed)

    # Group records by patient and determine majority label
    patient_records: dict[str, list[MSDRecord]] = defaultdict(list)
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
    logger.info("  Train: %d patients, %d slices (pos=%d neg=%d)",
                len(train_pids), len(train),
                sum(1 for r in train if r.label == 1),
                sum(1 for r in train if r.label == 0))
    logger.info("  Val:   %d patients, %d slices (pos=%d neg=%d)",
                len(val_pids), len(val),
                sum(1 for r in val if r.label == 1),
                sum(1 for r in val if r.label == 0))
    logger.info("  Test:  %d patients, %d slices (pos=%d neg=%d)",
                len(test_pids), len(test),
                sum(1 for r in test if r.label == 1),
                sum(1 for r in test if r.label == 0))

    return train, val, test


def write_csv(records: list[MSDRecord], path: Path) -> None:
    """Write MSD records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "label", "spacing_x", "spacing_y", "spacing_z",
        "patient_id", "tumor_pixels", "slice_has_tumor",
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
                "tumor_pixels": r.tumor_pixels,
                "slice_has_tumor": r.slice_has_tumor,
            })

    logger.info("Wrote %d records to %s", len(records), path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract MSD binary labels from NIfTI segmentation masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["msd-colon", "msd-hepatic-vessel"],
        help="MSD dataset name",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing raw MSD data (e.g., Task10_Colon)",
    )
    parser.add_argument(
        "--index-csv", type=Path, required=True,
        help="Path to preprocessed index CSV with spacing",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for train/val/test CSVs",
    )
    parser.add_argument(
        "--min-tumor-pixels", type=int, default=10,
        help="Minimum tumor pixels to consider a slice positive",
    )
    parser.add_argument(
        "--negative-ratio", type=float, default=1.0,
        help="Ratio of negative to positive slices to include",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    # Determine label directory and prefix based on dataset
    if args.dataset == "msd-colon":
        labels_dir = args.raw_dir / "labelsTr"
        dataset_prefix = "colon"
    elif args.dataset == "msd-hepatic-vessel":
        labels_dir = args.raw_dir / "labelsTr"
        dataset_prefix = "hepaticvessel"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info("Dataset: %s", args.dataset)
    logger.info("Labels directory: %s", labels_dir)
    logger.info("Dataset prefix: %s", dataset_prefix)

    logger.info("Loading index from %s", args.index_csv)
    series_map = load_index(args.index_csv)
    logger.info("Loaded %d series from index", len(series_map))

    records = extract_msd_records(
        series_map,
        labels_dir,
        dataset_prefix,
        min_tumor_pixels=args.min_tumor_pixels,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )

    if not records:
        logger.error("No records extracted! Check raw data and index CSV.")
        sys.exit(1)

    train, val, test = patient_stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Write CSVs
    dataset_name = args.dataset.replace("-", "_")
    write_csv(train, args.output_dir / f"{dataset_name}_train.csv")
    write_csv(val, args.output_dir / f"{dataset_name}_val.csv")
    write_csv(test, args.output_dir / f"{dataset_name}_test.csv")

    # Also write the full dataset for reference
    write_csv(records, args.output_dir / f"{dataset_name}_all.csv")

    logger.info("Done! Label extraction complete.")


if __name__ == "__main__":
    main()

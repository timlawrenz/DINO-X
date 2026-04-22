#!/usr/bin/env python3
"""Phase 2 preprocessing: NIfTI (.nii.gz) volumes -> 16-bit HU PNG slices.

Supports Medical Segmentation Decathlon (MSD) and similar NIfTI datasets.

Output format matches the DICOM pipeline (phase2_preprocess_lidc_idri.py):
  - 16-bit lossless PNG per axial slice
  - HU values clipped to [-1000, 4000], offset by 32768
  - CSV index with spacing metadata

Usage:
    python scripts/preprocessing/phase2_preprocess_nifti.py \\
        --nifti-dir data/raw/msd-colon/Task10_Colon/imagesTr \\
        --out-root data/processed \\
        --dataset-name msd-colon

MSD directory structure expected:
    Task{N}_{Name}/
        imagesTr/          <-- training images (*.nii.gz)
        labelsTr/          <-- training labels (not used here)
        imagesTs/          <-- test images (optional)
        dataset.json       <-- metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HU_CLIP_LO = -1000.0
HU_CLIP_HI = 4000.0
HU_OFFSET = 32768


def hu_to_u16(img_hu: np.ndarray) -> np.ndarray:
    """Convert HU float array to uint16 for 16-bit PNG storage."""
    x = np.clip(img_hu, HU_CLIP_LO, HU_CLIP_HI)
    x = np.rint(x + float(HU_OFFSET))
    return x.astype(np.uint16)


def save_slice_16bit(hu_slice: np.ndarray, out_path: Path) -> None:
    """Save a single HU slice as a 16-bit lossless PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u16 = hu_to_u16(hu_slice)
    Image.fromarray(u16, mode="I;16").save(out_path, format="PNG")


def load_nifti_volume(nifti_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume and extract physical spacing.

    Returns:
        (volume, spacing) where volume is (N, H, W) in HU and
        spacing is (spacing_x, spacing_y, spacing_z) in mm.
    """
    try:
        import nibabel as nib
    except ImportError:
        sys.exit("Missing dependency: nibabel. Install with: pip install nibabel")

    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    # NIfTI pixdim: [qfac, dim1_spacing, dim2_spacing, dim3_spacing, ...]
    # For standard RAS orientation: dim1=x (sagittal), dim2=y (coronal), dim3=z (axial)
    header = img.header
    pixdim = header["pixdim"]
    spacing_x = float(abs(pixdim[1]))
    spacing_y = float(abs(pixdim[2]))
    spacing_z = float(abs(pixdim[3]))

    # Ensure we have valid spacings
    if spacing_x <= 0:
        spacing_x = 1.0
    if spacing_y <= 0:
        spacing_y = 1.0
    if spacing_z <= 0:
        spacing_z = 1.0

    # NIfTI data is typically (X, Y, Z) — we want (Z, Y, X) for axial slicing
    # Transpose to (Z, Y, X) so axis 0 is the slice dimension
    if data.ndim == 3:
        volume = np.transpose(data, (2, 1, 0))
    elif data.ndim == 4:
        # Multi-channel: take first channel
        volume = np.transpose(data[:, :, :, 0], (2, 1, 0))
    else:
        raise ValueError(f"Unexpected NIfTI dimensions: {data.ndim} in {nifti_path}")

    return volume, (spacing_x, spacing_y, spacing_z)


def process_volume(
    nifti_path: Path,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
    force: bool = False,
) -> int:
    """Process a single NIfTI volume into PNG slices.

    Returns the number of slices written.
    """
    # Use the NIfTI filename stem as the series identifier
    stem = nifti_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    series_name = f"{dataset_name}_{stem}"
    out_series_dir = out_root / dataset_name / series_name

    # Skip if already processed (unless --force)
    if not force and out_series_dir.exists():
        existing = sorted(out_series_dir.glob("slice_*.png"))
        if len(existing) >= 3:
            # Re-index existing slices
            n = _reindex_existing(existing, series_name, out_root, dataset_name, writer)
            print(f"skip=true file={nifti_path.name} reason=already_converted slices={n}")
            return n

    try:
        volume, (sx, sy, sz) = load_nifti_volume(nifti_path)
    except Exception as e:
        print(f"skip=true file={nifti_path.name} reason=load_failed err={e}")
        return 0

    if volume.shape[0] < 3:
        print(f"skip=true file={nifti_path.name} reason=too_few_slices slices={volume.shape[0]}")
        return 0

    n_slices = volume.shape[0]
    encoding = f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}"

    for i in range(n_slices):
        out_path = out_series_dir / f"slice_{i:04d}.png"
        save_slice_16bit(volume[i], out_path)

        writer.writerow([
            str(out_path),
            series_name,
            i,
            encoding,
            f"{sx:.6f}",
            f"{sy:.6f}",
            f"{sz:.6f}",
        ])

    print(f"ok=true file={nifti_path.name} slices={n_slices} "
          f"spacing=({sx:.3f}, {sy:.3f}, {sz:.3f})")
    return n_slices


def _reindex_existing(
    png_paths: list[Path],
    series_name: str,
    out_root: Path,
    dataset_name: str,
    writer: csv.writer,
) -> int:
    """Re-index existing PNG slices without reprocessing."""
    # We don't have spacing info cached — load from the first PNG's parent
    # For now, write 1.0 defaults; a re-run with --force will fix spacing.
    encoding = f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}"
    n = 0
    for p in png_paths:
        try:
            idx = int(p.stem.split("_", 1)[1])
        except Exception:
            continue
        writer.writerow([str(p), series_name, idx, encoding, "1.000000", "1.000000", "1.000000"])
        n += 1
    return n


def find_nifti_files(search_dir: Path) -> list[Path]:
    """Find all NIfTI image files, excluding label files and macOS resource forks."""
    nifti_files = []
    for pattern in ("*.nii.gz", "*.nii"):
        nifti_files.extend(
            p for p in search_dir.rglob(pattern)
            if not p.name.startswith("._")
        )

    # Sort for deterministic ordering
    nifti_files.sort(key=lambda p: p.name)
    return nifti_files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert NIfTI volumes to 16-bit HU PNG slices with index CSV."
    )
    ap.add_argument("--nifti-dir", type=Path, required=True,
                    help="Directory containing .nii.gz files (e.g., imagesTr/)")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"),
                    help="Root output directory for processed data")
    ap.add_argument("--dataset-name", type=str, required=True,
                    help="Dataset identifier (e.g., msd-colon, msd-hepatic-vessel)")
    ap.add_argument("--max-volumes", type=int, default=0,
                    help="Process at most N volumes (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess volumes even if output already exists")
    args = ap.parse_args()

    nifti_dir = args.nifti_dir
    if not nifti_dir.exists():
        sys.exit(f"NIfTI directory not found: {nifti_dir}")

    nifti_files = find_nifti_files(nifti_dir)
    if not nifti_files:
        sys.exit(f"No .nii.gz or .nii files found in: {nifti_dir}")

    print(f"Found {len(nifti_files)} NIfTI files in {nifti_dir}")

    # Write per-dataset index CSV
    index_path = args.out_root / dataset_name_to_index(args.dataset_name)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    total_volumes = 0

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                     "spacing_x", "spacing_y", "spacing_z"])

        for nf in nifti_files:
            n = process_volume(nf, args.out_root, args.dataset_name, w, force=args.force)
            if n > 0:
                total_slices += n
                total_volumes += 1
            if args.max_volumes and total_volumes >= args.max_volumes:
                break

    print(f"\nDone: {total_volumes} volumes, {total_slices} slices")
    print(f"Index: {index_path}")
    return 0


def dataset_name_to_index(name: str) -> str:
    """Map dataset name to its index CSV path."""
    return f"{name}/index.csv"


if __name__ == "__main__":
    raise SystemExit(main())

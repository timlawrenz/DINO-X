#!/usr/bin/env python3
"""Phase 2 preprocessing: LIDC-IDRI DICOM -> 16-bit HU PNG slices.

Output encoding is **raw Hounsfield Units (HU)** stored as 16-bit lossless PNG
(one file per axial slice). We intentionally avoid baking fixed CT windows here;
windowing (including *random windowing*) is applied during training (Phase 3+).

Storage mapping (HU -> uint16 PNG):
- Clip HU to a reasonable CT range.
- Add an integer offset so values are non-negative.

For development without the real dataset, use --dry-run to generate a synthetic volume.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _need(mod: str) -> None:
    raise SystemExit(
        f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry."
    )


try:
    import numpy as np
except Exception:  # pragma: no cover
    _need("numpy")

try:
    from PIL import Image
except Exception:  # pragma: no cover
    _need("pillow")


HU_CLIP_LO = -1000.0
HU_CLIP_HI = 4000.0
HU_OFFSET = 32768


@dataclass(frozen=True)
class SliceKey:
    sort_z: float
    instance: int
    path: Path


def hu_to_u16(img_hu: "np.ndarray") -> "np.ndarray":
    x = np.clip(img_hu, HU_CLIP_LO, HU_CLIP_HI)
    x = np.rint(x + float(HU_OFFSET))
    return x.astype(np.uint16)


def save_slice_16bit(hu_slice: "np.ndarray", out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u16 = hu_to_u16(hu_slice)
    Image.fromarray(u16, mode="I;16").save(out_path, format="PNG")


@dataclass(frozen=True)
class SeriesSpacing:
    """Physical spacing extracted from DICOM headers."""

    spacing_x: float = 1.0
    spacing_y: float = 1.0
    spacing_z: float = 1.0


def _extract_spacing_from_dicom(ds) -> SeriesSpacing:
    """Extract PixelSpacing and SliceThickness from a pydicom Dataset."""
    ps = getattr(ds, "PixelSpacing", None)
    st = getattr(ds, "SliceThickness", None)
    sbs = getattr(ds, "SpacingBetweenSlices", None)
    return SeriesSpacing(
        spacing_x=float(ps[0]) if ps is not None and len(ps) >= 2 else 1.0,
        spacing_y=float(ps[1]) if ps is not None and len(ps) >= 2 else 1.0,
        spacing_z=float(sbs) if sbs is not None else (float(st) if st is not None else 1.0),
    )


def _load_dicom_series(series_dir: Path) -> tuple["np.ndarray", SeriesSpacing]:
    """Load a DICOM series as a 3D numpy volume and extract spacing.

    Computes z-spacing from ImagePositionPatient deltas when available
    (more reliable than SliceThickness). Falls back to
    SpacingBetweenSlices, then SliceThickness.

    Returns (volume, spacing) where volume is shape (N, H, W) in HU.
    """
    try:
        import pydicom  # type: ignore
        from pydicom.dataset import FileMetaDataset  # type: ignore
        from pydicom.uid import ImplicitVRLittleEndian  # type: ignore
    except Exception:  # pragma: no cover
        _need("pydicom")

    files: list[Path] = []
    for root, _, fnames in os.walk(series_dir):
        for f in fnames:
            fp = Path(root) / f
            # Many TCIA DICOMs have no extension; accept all regular files.
            if fp.is_file():
                files.append(fp)

    keys: list[SliceKey] = []
    header_spacing: SeriesSpacing | None = None

    for fp in files:
        try:
            # Note: stop_before_pixels=True means PixelData won't be present even for image slices.
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue

        # Heuristic: image slices typically have Rows/Columns.
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue

        instance = int(getattr(ds, "InstanceNumber", 0) or 0)
        ipp = getattr(ds, "ImagePositionPatient", None)
        sort_z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else float(instance)
        keys.append(SliceKey(sort_z=sort_z, instance=instance, path=fp))

        # Extract spacing from first valid DICOM header
        if header_spacing is None:
            header_spacing = _extract_spacing_from_dicom(ds)

    if not keys:
        raise RuntimeError(f"No readable DICOM image slices found under: {series_dir}")

    keys.sort(key=lambda k: (k.sort_z, k.instance, str(k.path)))

    # Compute z-spacing from IPP deltas (more reliable than SliceThickness)
    z_positions = sorted({k.sort_z for k in keys})
    if len(z_positions) >= 2:
        deltas = sorted(
            abs(z_positions[i + 1] - z_positions[i])
            for i in range(len(z_positions) - 1)
        )
        ipp_z = deltas[len(deltas) // 2]  # median
        if ipp_z > 0:
            sp = header_spacing or SeriesSpacing()
            header_spacing = SeriesSpacing(
                spacing_x=sp.spacing_x, spacing_y=sp.spacing_y, spacing_z=ipp_z,
            )

    spacing = header_spacing or SeriesSpacing()

    slices: list["np.ndarray"] = []
    base_shape: tuple[int, int] | None = None

    for k in keys:
        ds = pydicom.dcmread(str(k.path), force=True)

        # Some TCIA objects may omit Transfer Syntax in file_meta; assume implicit LE.
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = FileMetaDataset()
        if getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        if "PixelData" not in ds:
            continue

        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            # If compressed pixel data handlers are missing, skip and let the user
            # install additional decoders (pylibjpeg/gdcm) if needed.
            continue

        if base_shape is None:
            base_shape = (int(arr.shape[0]), int(arr.shape[1]))
        elif base_shape != (int(arr.shape[0]), int(arr.shape[1])):
            continue

        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        hu = arr * slope + intercept
        slices.append(hu)

    if not slices:
        raise RuntimeError(
            f"No decodable pixel slices found under: {series_dir}. "
            "If the data is compressed, install an image decoder (e.g., pylibjpeg/gdcm)."
        )

    vol = np.stack(slices, axis=0)
    return vol, spacing


def _synthetic_volume(n: int = 16, h: int = 128, w: int = 128) -> "np.ndarray":
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (xx / (w - 1) * 800.0) - 600.0  # ~HU-ish gradient
    vol = []
    for i in range(n):
        z = (i - n / 2.0) / (n / 2.0)
        blob = 400.0 * np.exp(-(((xx - w * 0.6) ** 2 + (yy - h * 0.4) ** 2) / (2 * (w * 0.08) ** 2)))
        vol.append(base + blob + z * 50.0)
    return np.stack(vol, axis=0)


def iter_series_dirs(dicom_root: Path) -> Iterable[Path]:
    # Minimal heuristic: treat any directory containing >=1 file as a "series".
    # This is intentionally simple; LIDC raw layouts vary by acquisition method.
    for root, dirnames, fnames in os.walk(dicom_root):
        if fnames:
            yield Path(root)
            # Don’t recurse further once we find files in this subtree.
            dirnames[:] = []


def _quick_extract_spacing(series_dir: Path) -> SeriesSpacing:
    """Read one DICOM header from a series dir for spacing (no pixel data)."""
    try:
        import pydicom  # type: ignore
    except ImportError:
        return SeriesSpacing()

    for root, _, fnames in os.walk(series_dir):
        for fname in sorted(fnames):
            fp = Path(root) / fname
            if not fp.is_file() or fp.name in ("LICENSE", ".complete"):
                continue
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if hasattr(ds, "Rows"):
                return _extract_spacing_from_dicom(ds)
    return SeriesSpacing()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert DICOM series to 16-bit HU PNG slices with index CSV."
    )
    ap.add_argument("--dicom-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--dataset-name", type=str, default="lidc-idri",
                    help="Name for the output subdirectory (e.g., lidc-idri, pancreas-ct, cq500)")
    ap.add_argument("--index-path", type=Path, default=None,
                    help="Output index CSV path (default: {out-root}/{dataset-name}/index.csv)")
    ap.add_argument("--max-series", type=int, default=0, help="0 = no limit")
    ap.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Re-generate PNG slices even if the output series folder already exists",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root
    dataset_name = args.dataset_name
    if args.index_path is not None:
        index_path = args.index_path
    else:
        # Per-dataset index: each dataset gets its own index.csv
        index_path = out_root / dataset_name / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding",
                     "spacing_x", "spacing_y", "spacing_z"])

        if args.dry_run:
            vol = _synthetic_volume()
            series_name = "synthetic"
            spacing = SeriesSpacing(spacing_x=0.75, spacing_y=0.75, spacing_z=1.5)
            _write_volume(vol, series_name, out_root, w,
                          dataset_name=dataset_name, spacing=spacing)
            print("ok=true")
            print(f"index={index_path}")
            return 0

        dicom_root = args.dicom_root
        if not dicom_root.exists():
            raise SystemExit(f"dicom_root not found: {dicom_root}")

        n_processed = 0
        for series_dir in iter_series_dirs(dicom_root):
            if series_dir.name.startswith("_"):
                continue

            series_name = series_dir.as_posix().replace("/", "_")
            out_series_dir = out_root / dataset_name / series_name

            # Default incremental behavior: if the output folder already exists, skip
            # PNG generation and only re-index what's already on disk.
            # Still extract spacing from DICOM headers for the index CSV.
            if not args.force_reprocess and out_series_dir.exists():
                skip_spacing = _quick_extract_spacing(series_dir)
                n = _write_existing_series(
                    out_series_dir, w, series_dir=series_dir, spacing=skip_spacing,
                )
                if n >= 3:
                    print(f"skip=true series_dir={series_dir} reason=already_converted slices={n}")
                    n_processed += 1
                    if args.max_series and n_processed >= args.max_series:
                        break
                    continue

            try:
                vol, spacing = _load_dicom_series(series_dir)
            except RuntimeError as e:
                print(f"skip=true series_dir={series_dir} reason=load_failed err={e}")
                continue

            if vol.shape[0] < 3:
                print(
                    f"skip=true series_dir={series_dir} reason=too_few_slices slices={vol.shape[0]}"
                )
                continue

            _write_volume(vol, series_name, out_root, w,
                          series_dir=series_dir, dataset_name=dataset_name, spacing=spacing)
            n_processed += 1
            if args.max_series and n_processed >= args.max_series:
                break

    print("ok=true")
    print(f"index={index_path}")
    return 0


def _write_existing_series(
    out_series_dir: Path,
    writer: "csv.writer",
    *,
    series_dir: Optional[Path] = None,
    spacing: SeriesSpacing | None = None,
) -> int:
    # Re-index existing PNG slices from disk.
    sp = spacing or SeriesSpacing()
    paths = sorted(out_series_dir.glob("slice_*.png"))
    n = 0
    for p in paths:
        stem = p.stem  # slice_0001
        try:
            idx = int(stem.split("_", 1)[1])
        except Exception:
            continue
        writer.writerow(
            [
                str(p),
                str(series_dir) if series_dir is not None else out_series_dir.name,
                idx,
                f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}",
                f"{sp.spacing_x:.6f}",
                f"{sp.spacing_y:.6f}",
                f"{sp.spacing_z:.6f}",
            ]
        )
        n += 1
    return n


def _write_volume(
    vol: "np.ndarray",
    series_name: str,
    out_root: Path,
    writer: "csv.writer",
    series_dir: Optional[Path] = None,
    dataset_name: str = "lidc-idri",
    spacing: SeriesSpacing | None = None,
) -> None:
    sp = spacing or SeriesSpacing()
    # Raw HU slices: one 16-bit grayscale PNG per axial slice.
    for i in range(vol.shape[0]):
        out_path = out_root / dataset_name / series_name / f"slice_{i:04d}.png"
        save_slice_16bit(vol[i], out_path)

        writer.writerow([
            str(out_path),
            str(series_dir) if series_dir is not None else series_name,
            i,
            f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}",
            f"{sp.spacing_x:.6f}",
            f"{sp.spacing_y:.6f}",
            f"{sp.spacing_z:.6f}",
        ])


if __name__ == "__main__":
    raise SystemExit(main())

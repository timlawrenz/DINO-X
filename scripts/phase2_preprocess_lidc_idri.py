#!/usr/bin/env python3
"""Phase 2 preprocessing: LIDC-IDRI DICOM -> 2.5D RGB PNG.

Default output encoding is a *hybrid* 2.5D+windowing representation:
- R channel: slice i-1 windowed with Lung settings
- G channel: slice i windowed with Soft-Tissue settings
- B channel: slice i+1 windowed with Bone settings

This satisfies both:
- "3 depth slices per image" (2.5D)
- "Lung/Soft-Tissue/Bone" windowing (RGB)

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


Window = tuple[float, float]  # (center, width)

LUNG: Window = (-600.0, 1500.0)
SOFT: Window = (40.0, 400.0)
BONE: Window = (300.0, 1500.0)


@dataclass(frozen=True)
class SliceKey:
    sort_z: float
    instance: int
    path: Path


def window_to_u8(img_hu: "np.ndarray", center: float, width: float) -> "np.ndarray":
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).astype(np.uint8)


def save_rgb(rgb: "np.ndarray", out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def _load_dicom_series(series_dir: Path) -> "np.ndarray":
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

    if not keys:
        raise RuntimeError(f"No readable DICOM image slices found under: {series_dir}")

    keys.sort(key=lambda k: (k.sort_z, k.instance, str(k.path)))

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
    return vol


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--max-series", type=int, default=0, help="0 = no limit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root
    index_path = out_root / "_index" / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["png_path", "series_dir", "slice_index", "encoding"])  # minimal

        if args.dry_run:
            vol = _synthetic_volume()
            series_name = "synthetic"
            _write_volume(vol, series_name, out_root, w)
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

            try:
                vol = _load_dicom_series(series_dir)
            except RuntimeError as e:
                print(f"skip=true series_dir={series_dir} reason=load_failed err={e}")
                continue

            if vol.shape[0] < 3:
                print(
                    f"skip=true series_dir={series_dir} reason=too_few_slices slices={vol.shape[0]}"
                )
                continue

            series_name = series_dir.as_posix().replace("/", "_")
            _write_volume(vol, series_name, out_root, w, series_dir=series_dir)
            n_processed += 1
            if args.max_series and n_processed >= args.max_series:
                break

    print("ok=true")
    print(f"index={index_path}")
    return 0


def _write_volume(
    vol: "np.ndarray",
    series_name: str,
    out_root: Path,
    writer: "csv.writer",
    series_dir: Optional[Path] = None,
) -> None:
    # Hybrid encoding: (lung i-1, soft i, bone i+1)
    for i in range(1, vol.shape[0] - 1):
        r = window_to_u8(vol[i - 1], *LUNG)
        g = window_to_u8(vol[i], *SOFT)
        b = window_to_u8(vol[i + 1], *BONE)
        rgb = np.stack([r, g, b], axis=-1)

        out_path = out_root / "lidc-idri" / series_name / f"slice_{i:04d}.png"
        save_rgb(rgb, out_path)

        writer.writerow([
            str(out_path),
            str(series_dir) if series_dir is not None else series_name,
            i,
            "hybrid_depth+window",
        ])


if __name__ == "__main__":
    raise SystemExit(main())

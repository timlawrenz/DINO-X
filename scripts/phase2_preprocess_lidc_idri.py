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
    ap.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Re-generate PNG slices even if the output series folder already exists",
    )
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

            series_name = series_dir.as_posix().replace("/", "_")
            out_series_dir = out_root / "lidc-idri" / series_name

            # Default incremental behavior: if the output folder already exists, skip
            # PNG generation and only re-index what's already on disk.
            if not args.force_reprocess and out_series_dir.exists():
                n = _write_existing_series(out_series_dir, w, series_dir=series_dir)
                if n >= 3:
                    print(f"skip=true series_dir={series_dir} reason=already_converted slices={n}")
                    n_processed += 1
                    if args.max_series and n_processed >= args.max_series:
                        break
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

            _write_volume(vol, series_name, out_root, w, series_dir=series_dir)
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
) -> int:
    # Re-index existing PNG slices from disk.
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
) -> None:
    # Raw HU slices: one 16-bit grayscale PNG per axial slice.
    for i in range(vol.shape[0]):
        out_path = out_root / "lidc-idri" / series_name / f"slice_{i:04d}.png"
        save_slice_16bit(vol[i], out_path)

        writer.writerow([
            str(out_path),
            str(series_dir) if series_dir is not None else series_name,
            i,
            f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}",
        ])


if __name__ == "__main__":
    raise SystemExit(main())

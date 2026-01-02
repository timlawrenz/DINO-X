#!/usr/bin/env python3
"""Phase 2 validation: sample a few processed PNGs and make a contact sheet."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


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


HU_OFFSET = 32768.0


def _window_hu_to_u8(img_hu: "np.ndarray", level: float, width: float) -> "np.ndarray":
    lo = level - width / 2.0
    hi = level + width / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).astype(np.uint8)


def _preview_rgb(p: Path) -> Image.Image:
    im = Image.open(p)
    if im.mode.startswith("I"):
        arr = np.asarray(im)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        hu = arr.astype(np.float32) - HU_OFFSET
        u8 = _window_hu_to_u8(hu, level=-600.0, width=1500.0)  # lung-ish preview
        return Image.fromarray(u8, mode="L").convert("RGB")
    return im.convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/_validation"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pngs = sorted(args.processed_root.rglob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found under: {args.processed_root}")

    picks = rng.sample(pngs, k=min(args.n, len(pngs)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [_preview_rgb(p) for p in picks]

    # Save individual picks
    for i, (p, im) in enumerate(zip(picks, imgs)):
        out = args.out_dir / f"sample_{i:02d}{p.suffix}"
        im.save(out)

    # Contact sheet
    cols = 5
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for idx, im in enumerate(imgs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        sheet.paste(im, (x, y))

    sheet_path = args.out_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    print("ok=true")
    print(f"out_dir={args.out_dir}")
    print(f"contact_sheet={sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

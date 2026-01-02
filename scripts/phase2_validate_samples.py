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
    from PIL import Image
except Exception:  # pragma: no cover
    _need("pillow")


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

    imgs = [Image.open(p).convert("RGB") for p in picks]

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

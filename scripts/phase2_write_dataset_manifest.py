#!/usr/bin/env python3
"""Write a small, non-PHI dataset manifest for reproducibility.

This is intentionally lightweight: it records counts/sizes and layout, not patient metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DirStats:
    path: str
    files: int
    bytes: int


def dir_stats(p: Path) -> DirStats:
    files = 0
    total = 0
    if not p.exists():
        return DirStats(path=str(p), files=0, bytes=0)
    for root, _, fnames in os.walk(p):
        for f in fnames:
            fp = Path(root) / f
            try:
                st = fp.stat()
            except FileNotFoundError:
                continue
            files += 1
            total += st.st_size
    return DirStats(path=str(p), files=files, bytes=total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = {
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw": asdict(dir_stats(args.raw_root)),
        "processed": asdict(dir_stats(args.processed_root)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("ok=true")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

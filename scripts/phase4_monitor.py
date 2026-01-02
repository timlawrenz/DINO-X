#!/usr/bin/env python3
"""Phase 4: Training monitor (run safely alongside training).

This script is intentionally read-only with respect to training state: it loads a
checkpoint, runs a small forward pass on deterministic samples, and writes
visualizations/metrics to a timestamped output directory.

Note: Phase 3's micro-run model does not expose true self-attention weights.
For Phase 4 we emit a stable *proxy* "attention" heatmap derived from patch-token
magnitudes (useful as a day-over-day drift/collapse signal).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(
        f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry. "
        "If you're using ROCm PyTorch, ensure ROCm libs are discoverable (e.g., `source scripts/rocm_env.*`)."
    )


try:
    import numpy as np
except Exception:  # pragma: no cover
    _need("numpy")

try:
    from PIL import Image
except Exception:  # pragma: no cover
    _need("pillow")

try:
    import torch
except ImportError:  # pragma: no cover
    _need("torch")


def _load_phase3_module() -> Any:
    p = Path(__file__).resolve().parent / "phase3_micro_run.py"
    spec = importlib.util.spec_from_file_location("phase3_micro_run", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec to keep dataclasses happy.
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _load_split_series_dirs(split_manifest: Path, which: str) -> set[str]:
    if which not in {"train", "val"}:
        raise SystemExit("--split must be one of: train, val")
    payload = json.loads(split_manifest.read_text())
    series = payload.get(which, {}).get("series_dir", [])
    if not isinstance(series, list):
        raise SystemExit(f"Invalid split manifest (missing {which}.series_dir): {split_manifest}")
    return set(str(s) for s in series)


def _load_fixed_sample_tensor(
    m: Any,
    rows: list[Any],
    *,
    row: Any,
    img_size: int,
    level: float,
    width: float,
) -> torch.Tensor:
    # Mirrors PngDataset HU16 path, but with fixed windowing (no randomness).
    png_path: Path = Path(row.png_path)

    # Determine if HU16 (vs baked RGB) using the same heuristic as Phase 3.
    try:
        mode = Image.open(png_path).mode
    except Exception:
        mode = ""

    is_hu16 = str(row.encoding).startswith("hu16") or mode.startswith("I")
    if not is_hu16:
        im = Image.open(png_path).convert("RGB")
        if im.size != (img_size, img_size):
            im = im.resize((img_size, img_size), resample=Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    # Build a quick series map (slice_index -> png_path) for the sample's series.
    series_dir = str(row.series_dir)
    mp: dict[int, Path] = {}
    for r in rows:
        if str(r.series_dir) == series_dir:
            mp[int(r.slice_index)] = Path(r.png_path)

    if not mp:
        raise SystemExit(f"series_dir not found in index rows: {series_dir}")

    ks = sorted(mp.keys())
    z0, z1 = ks[0], ks[-1]

    def _clamp(k: int) -> int:
        return max(z0, min(z1, k))

    z = int(row.slice_index)
    p_m1 = mp.get(_clamp(z - 1), png_path)
    p_0 = mp.get(_clamp(z), png_path)
    p_p1 = mp.get(_clamp(z + 1), png_path)

    def _load_hu01(p: Path) -> np.ndarray:
        im = Image.open(p)
        if im.size != (img_size, img_size):
            im = im.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(im)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        hu = arr.astype(np.float32) - float(m.HU_OFFSET)
        return m._window_hu_to_01(hu, level=level, width=width)

    a = _load_hu01(p_m1)
    b = _load_hu01(p_0)
    c = _load_hu01(p_p1)

    x = np.stack([a, b, c], axis=0)
    return torch.from_numpy(x).contiguous()


def _write_heatmap_png(heat: np.ndarray, out: Path, *, size: int) -> None:
    # heat: (gh, gw) in [0,1]
    img = Image.fromarray((np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    img = img.resize((size, size), resample=Image.BILINEAR)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument("--split", choices=["train", "val"], default="train")

    ap.add_argument("--fixed-png", type=Path, default=None, help="Optional explicit sample png_path")
    ap.add_argument("--sample-seed", type=int, default=0, help="Used when --fixed-png is not set")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batch-seed", type=int, default=0)

    ap.add_argument("--level", type=float, default=-600.0, help="Fixed HU window level")
    ap.add_argument("--width", type=float, default=1500.0, help="Fixed HU window width")

    ap.add_argument("--out-root", type=Path, default=Path("data/monitor/phase4"))
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    m = _load_phase3_module()

    # NOTE: local/trusted checkpoint (stores config + rng).
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = payload.get("config") or {}

    img_size = int(ckpt_cfg.get("img_size", 224))
    patch = int(ckpt_cfg.get("patch", 16))
    dim = int(ckpt_cfg.get("dim", 384))
    depth = int(ckpt_cfg.get("depth", 6))
    heads = int(ckpt_cfg.get("heads", 6))
    mlp_ratio = float(ckpt_cfg.get("mlp_ratio", 4.0))
    out_dim = int(ckpt_cfg.get("out_dim", 8192))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = m.PatchViT(img_size=img_size, patch=patch, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)
    model = m.DinoStudentTeacher(vit, out_dim=out_dim).to(device)
    model.load_state_dict(payload["student"], strict=True)
    model.eval()

    rows = m._load_index_rows(args.index_csv)

    if args.split_manifest is not None:
        keep = _load_split_series_dirs(args.split_manifest, args.split)
        rows = [r for r in rows if str(r.series_dir) in keep]

    if not rows:
        raise SystemExit("No rows available after applying split filter")

    # Pick fixed sample.
    fixed_row = None
    if args.fixed_png is not None:
        for r in rows:
            if Path(r.png_path) == args.fixed_png:
                fixed_row = r
                break
        if fixed_row is None:
            raise SystemExit(f"--fixed-png not found in filtered index: {args.fixed_png}")
    else:
        rr = rows[:]
        rng = random.Random(args.sample_seed)
        rng.shuffle(rr)
        fixed_row = rr[0]

    # Output directory.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed-sample proxy attention.
    x = _load_fixed_sample_tensor(m, rows, row=fixed_row, img_size=img_size, level=args.level, width=args.width)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        cls, patches = model.vit.forward_features(x)

    # Heatmap: patch token L2 norm reshaped to patch grid.
    p = patches[0]  # (P, D)
    scores = torch.norm(p, dim=-1)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    grid = img_size // patch
    heat = scores.reshape(grid, grid).detach().cpu().numpy().astype(np.float32)
    heat_path = out_dir / "attention_heatmap.png"
    _write_heatmap_png(heat, heat_path, size=img_size)

    # Embedding dispersion on fixed batch.
    rr = rows[:]
    rng = random.Random(args.batch_seed)
    rng.shuffle(rr)
    rr = rr[: min(args.batch_size, len(rr))]

    embs: list[torch.Tensor] = []
    for r in rr:
        xb = _load_fixed_sample_tensor(m, rows, row=r, img_size=img_size, level=args.level, width=args.width)
        xb = xb.unsqueeze(0).to(device)
        with torch.no_grad():
            c, _p = model.vit.forward_features(xb)
        embs.append(c.squeeze(0).detach().cpu())

    E = torch.stack(embs, dim=0)  # (B, D)
    std_per_dim = E.std(dim=0)

    metrics = {
        "kind": "training_monitor",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": int(payload.get("step", 0) or 0),
        "device": str(device),
        "fixed_sample": {
            "png_path": str(fixed_row.png_path),
            "series_dir": str(fixed_row.series_dir),
            "slice_index": int(fixed_row.slice_index),
        },
        "batch": {
            "size": int(len(rr)),
            "seed": int(args.batch_seed),
            "split": args.split,
            "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        },
        "window": {"level": float(args.level), "width": float(args.width)},
        "embedding_dispersion": {
            "std_mean": float(std_per_dim.mean().item()),
            "std_min": float(std_per_dim.min().item()),
            "std_max": float(std_per_dim.max().item()),
        },
        "artifacts": {"attention_heatmap_png": str(heat_path)},
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Persist config inputs for reproducibility.
    cfg = {
        "checkpoint": str(args.checkpoint),
        "index_csv": str(args.index_csv),
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "split": args.split,
        "fixed_png": str(args.fixed_png) if args.fixed_png is not None else None,
        "sample_seed": args.sample_seed,
        "batch_size": args.batch_size,
        "batch_seed": args.batch_seed,
        "level": args.level,
        "width": args.width,
        "out_dir": str(out_dir),
        "ckpt_model_config": ckpt_cfg,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print("ok=true")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

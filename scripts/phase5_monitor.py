#!/usr/bin/env python3
"""Phase 5: Training monitor (run safely alongside training).

Generates "attention" heatmaps (magnitude of patch tokens) and checks embedding statistics
from a Phase 5 checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(
        f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry."
    )


try:
    import numpy as np
except Exception:
    _need("numpy")

try:
    from PIL import Image
except Exception:
    _need("pillow")

try:
    import torch
except ImportError:
    _need("torch")


def _load_phase5_module() -> Any:
    p = Path(__file__).resolve().parent / "phase5_big_run.py"
    spec = importlib.util.spec_from_file_location("phase5_big_run", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _load_fixed_sample_tensor(
    rows: list[Any],
    row: Any,
    img_size: int,
    level: float,
    width: float,
) -> torch.Tensor:
    # Re-implement 3-slice loading logic from Phase 5
    series_dir = str(row.series_dir)
    slice_idx = int(row.slice_index)
    
    # Build series map for this series only
    mp: dict[int, Path] = {}
    for r in rows:
        if str(r.series_dir) == series_dir:
            mp[int(r.slice_index)] = Path(r.png_path)
    
    if not mp:
        # Fallback if filtering removed context (shouldn't happen with proper logic)
        mp[slice_idx] = Path(row.png_path)

    ks = sorted(mp.keys())
    z0, z1 = ks[0], ks[-1]

    def _clamp(k: int) -> int:
        return max(z0, min(z1, k))

    paths = [
        mp.get(_clamp(slice_idx - 1), Path(row.png_path)),
        mp.get(_clamp(slice_idx), Path(row.png_path)),
        mp.get(_clamp(slice_idx + 1), Path(row.png_path)),
    ]

    def _load_hu01(p: Path) -> np.ndarray:
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        
        wmin = level - width / 2.0
        wmax = level + width / 2.0
        windowed = (hu - wmin) / max(width, 1.0)
        windowed = np.clip(windowed, 0.0, 1.0)
        
        if windowed.shape != (img_size, img_size):
            img_pil = Image.fromarray((windowed * 255).astype(np.uint8))
            img_pil = img_pil.resize((img_size, img_size), Image.Resampling.BILINEAR)
            windowed = np.array(img_pil, dtype=np.float32) / 255.0
        
        return windowed

    slices = [_load_hu01(p) for p in paths]
    x = np.stack(slices, axis=0)  # (3, H, W)
    return torch.from_numpy(x).contiguous()


def _write_heatmap_png(heat: np.ndarray, out: Path, size: int) -> None:
    # heat: (gh, gw) in [0,1]
    img = Image.fromarray((np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    img = img.resize((size, size), resample=Image.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 Monitor: Checkpoints & Heatmaps")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding stats")
    ap.add_argument("--fixed-png", type=Path, help="Specific PNG to visualize")
    ap.add_argument("--sample-seed", type=int, default=42, help="Seed for random sample selection")
    
    # Windowing for visualization (fixed)
    ap.add_argument("--level", type=float, default=-600.0)
    ap.add_argument("--width", type=float, default=1500.0)
    
    ap.add_argument("--out-dir", type=Path, default=Path("data/monitor/phase5"))
    
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load Phase 5 module dynamically
    m = _load_phase5_module()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint payload (trusted)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # Reconstruct config
    config_dict = payload.get("config", {})
    model_cfg_dict = config_dict.get("model", {})
    
    # Handle both object and dict (depending on how it was saved/loaded)
    if isinstance(model_cfg_dict, m.ModelConfig):
        model_cfg = model_cfg_dict
    else:
        model_cfg = m.ModelConfig(**model_cfg_dict)

    print(f"Model: {model_cfg.name} (patch={model_cfg.patch}, dim={model_cfg.dim})")

    # Initialize model
    vit = m.PatchViT(
        img_size=config_dict.get("img_size", 224),
        patch=model_cfg.patch,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_ratio=model_cfg.mlp_ratio,
        use_grad_checkpoint=False, # Eval mode
    )
    student = m.DinoStudentTeacher(vit, out_dim=model_cfg.out_dim).to(device)
    student.load_state_dict(payload["student"], strict=True)
    student.eval()

    # Load data index
    all_rows = m._load_index_rows(args.index_csv)
    print(f"Loaded {len(all_rows)} rows from index")

    # Select sample for heatmap
    if args.fixed_png:
        target_row = next((r for r in all_rows if Path(r.png_path) == args.fixed_png), None)
        if not target_row:
            raise ValueError(f"PNG not found in index: {args.fixed_png}")
    else:
        rng = random.Random(args.sample_seed)
        target_row = rng.choice(all_rows)

    print(f"Visualizing sample: {target_row.png_path}")
    
    # Load tensor
    img_size = config_dict.get("img_size", 224)
    x = _load_fixed_sample_tensor(
        all_rows, target_row, img_size, args.level, args.width
    ).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        # PatchViT forward returns (cls_token, patch_tokens) in recent PyTorch ViT? 
        # Wait, let's check Phase 5 PatchViT implementation.
        # It calls self.blocks... x = self.norm(x).
        # It returns x (B, N+1, D).
        
        feats = student.backbone(x)
        
    # Heatmap: L2 norm of patch tokens (skip CLS)
    patch_tokens = feats[:, 1:, :] # (1, N, D)
    norms = torch.norm(patch_tokens, dim=-1).squeeze(0) # (N,)
    
    # Normalize to [0, 1]
    norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    
    # Reshape to grid
    grid_size = int(np.sqrt(norms.shape[0]))
    heatmap = norms.reshape(grid_size, grid_size).cpu().numpy()
    
    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = args.out_dir / f"{timestamp}_step{payload.get('step', 0)}"
    run_out.mkdir(parents=True, exist_ok=True)
    
    heatmap_path = run_out / "heatmap.png"
    _write_heatmap_png(heatmap, heatmap_path, img_size)
    print(f"Saved heatmap to: {heatmap_path}")
    
    # Save original input slice (middle channel) for reference
    input_slice = x[0, 1, :, :].cpu().numpy() # Middle slice (z)
    input_path = run_out / "input.png"
    Image.fromarray((input_slice * 255).astype(np.uint8)).save(input_path)
    print(f"Saved input reference to: {input_path}")

    # Embedding dispersion stats
    print("Computing embedding stats...")
    rng = random.Random(args.sample_seed)
    batch_rows = rng.sample(all_rows, min(args.batch_size, len(all_rows)))
    
    embeddings = []
    for r in batch_rows:
        xt = _load_fixed_sample_tensor(
            all_rows, r, img_size, args.level, args.width
        ).unsqueeze(0).to(device)
        with torch.no_grad():
             f = student.backbone(xt)
             cls = f[:, 0, :] # CLS token
             embeddings.append(cls.cpu())
    
    E = torch.cat(embeddings, dim=0) # (B, D)
    std = E.std(dim=0).mean().item()
    norm = E.norm(dim=-1).mean().item()
    
    stats = {
        "step": payload.get("step", 0),
        "embedding_std_mean": std,
        "embedding_norm_mean": norm,
        "sample": str(target_row.png_path)
    }
    
    stats_path = run_out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats: std={std:.4f}, norm={norm:.4f}")

    return 0

if __name__ == "__main__":
    main()

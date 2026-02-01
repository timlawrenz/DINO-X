#!/usr/bin/env python3
"""Phase 5: Label-free view-retrieval evaluation.

Goal: objectively validate CT representations without labels.

Protocol:
- Sample N rows from the eval split (val.series_dir from split manifest).
- For each row, generate two views using Phase 5's PngDataset (same augmentations).
- Embed both views with the backbone (CLS token, pre-projection head).
- Compute retrieval accuracy: for query i (view1), does nearest key in view2 match i?

Outputs:
- Prints ok=true, passed=true|false and metrics.
- Writes a JSON metrics file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry.")


try:
    import numpy as np
except Exception:
    _need("numpy")

try:
    import torch
    import torch.nn.functional as F
except Exception:
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


def _load_split_series_dirs(split_manifest: Path, which: str) -> set[str]:
    payload = json.loads(split_manifest.read_text())
    series = payload.get(which, {}).get("series_dir", [])
    if not isinstance(series, list) or not series:
        raise SystemExit(f"Invalid split manifest (missing {which}.series_dir): {split_manifest}")
    return set(str(s) for s in series)


@torch.no_grad()
def _embed_backbone_cls(student: Any, x: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized CLS embedding from backbone."""
    feats = student.backbone(x)  # (B, N+1, D)
    cls = feats[:, 0]
    return F.normalize(cls.float(), p=2, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 label-free view-retrieval eval")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4096, help="Number of samples in eval set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: next to checkpoint (view_retrieval_step<step>_N<n>.json)",
    )
    ap.add_argument("--topk", type=int, default=5, help="Also compute top-k accuracy")
    ap.add_argument(
        "--ratio", type=float, default=10.0, help="Pass gate: top1 >= ratio*(1/N)"
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {args.index_csv}")
    if not args.split_manifest.exists():
        raise FileNotFoundError(f"split_manifest not found: {args.split_manifest}")
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.topk <= 0:
        raise SystemExit("--topk must be > 0")

    # Ensure deterministic sampling + view generation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    m = _load_phase5_module()

    device = None
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(payload.get("step", 0) or 0)
    cfg = payload.get("config", {})
    model_cfg = cfg.get("model", {})

    # Build model (student only)
    if isinstance(model_cfg, m.ModelConfig):
        mc = model_cfg
    else:
        mc = m.ModelConfig(**model_cfg)

    img_size = int(cfg.get("img_size", 224))

    vit = m.PatchViT(
        img_size=img_size,
        patch=mc.patch,
        dim=mc.dim,
        depth=mc.depth,
        heads=mc.heads,
        mlp_ratio=mc.mlp_ratio,
        use_grad_checkpoint=False,
    )
    student = m.DinoStudentTeacher(vit, out_dim=mc.out_dim).to(device)
    student.load_state_dict(payload["student"], strict=True)
    student.eval()

    # Load rows, filter to val series
    val_set = _load_split_series_dirs(args.split_manifest, "val")
    all_rows = m._load_index_rows(args.index_csv)
    rows = [r for r in all_rows if str(r.series_dir) in val_set]
    if not rows:
        raise SystemExit("No rows remain after filtering to val.series_dir")

    if len(rows) < args.n:
        print(
            f"⚠️  Requested --n={args.n} but only {len(rows)} val rows available; capping n."  # noqa: T201
        )
        args.n = len(rows)

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(rows)), k=args.n)

    # Build dataset over all val rows so 3-slice context (z-1,z,z+1) matches training.
    ds = m.PngDataset(
        rows,
        img_size=img_size,
        rw_level_min=float(cfg.get("rw_level_min", -400.0)),
        rw_level_max=float(cfg.get("rw_level_max", 400.0)),
        rw_width_min=float(cfg.get("rw_width_min", 800.0)),
        rw_width_max=float(cfg.get("rw_width_max", 2000.0)),
    )

    # Materialize views + embeddings in batches
    t0 = time.time()
    Q_chunks: list[torch.Tensor] = []
    K_chunks: list[torch.Tensor] = []

    bs = args.batch_size
    for start in range(0, args.n, bs):
        end = min(args.n, start + bs)

        v1_list = []
        v2_list = []
        for j in range(start, end):
            v1, v2 = ds[idxs[j]]
            v1_list.append(v1)
            v2_list.append(v2)

        x1 = torch.stack(v1_list, dim=0).to(device, non_blocking=True)
        x2 = torch.stack(v2_list, dim=0).to(device, non_blocking=True)

        q = _embed_backbone_cls(student, x1)
        k = _embed_backbone_cls(student, x2)

        Q_chunks.append(q.cpu())
        K_chunks.append(k.cpu())

    Q = torch.cat(Q_chunks, dim=0)  # (N, D)
    K = torch.cat(K_chunks, dim=0)  # (N, D)

    # Similarity matrix (N,N)
    # Use float32 on CPU to avoid fp16 weirdness.
    S = (Q.float() @ K.float().T).numpy()

    # Top-1
    top1_idx = np.argmax(S, axis=1)
    top1 = float(np.mean(top1_idx == np.arange(args.n)))

    # Top-k
    k = int(args.topk)
    if k >= args.n:
        k = args.n
    topk_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    topk = float(np.mean([(i in topk_idx[i]) for i in range(args.n)]))

    baseline = 1.0 / float(args.n)
    ratio = top1 / baseline if baseline > 0 else float("inf")
    passed = top1 >= float(args.ratio) * baseline

    dt = time.time() - t0

    out = args.out
    if out is None:
        out = args.checkpoint.parent / f"view_retrieval_step{step}_N{args.n}.json"

    metrics = {
        "kind": "phase5_view_retrieval",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": step,
        "index_csv": str(args.index_csv),
        "split_manifest": str(args.split_manifest),
        "img_size": img_size,
        "n": args.n,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "topk": int(args.topk),
        "top1": top1,
        "topk_acc": topk,
        "random_baseline": baseline,
        "ratio_vs_random": ratio,
        "pass_ratio": float(args.ratio),
        "passed": bool(passed),
        "seconds": dt,
        "model": {
            "name": mc.name,
            "patch": mc.patch,
            "dim": mc.dim,
            "depth": mc.depth,
            "heads": mc.heads,
            "mlp_ratio": mc.mlp_ratio,
            "out_dim": mc.out_dim,
            "ln_out_dim": math.log(float(mc.out_dim)),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    print("ok=true")
    print(f"passed={str(passed).lower()}")
    print(f"top1={top1:.6f} top{args.topk}={topk:.6f} baseline={baseline:.6f} ratio={ratio:.2f} seconds={dt:.1f}")
    print(f"metrics_json={out}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

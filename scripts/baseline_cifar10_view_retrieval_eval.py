#!/usr/bin/env python3
"""Baseline: CIFAR-10 label-free view-retrieval evaluation.

Protocol:
- Sample N images from CIFAR-10 (train or test split).
- Create two augmented views per image (same transforms as pretrain).
- Embed with the backbone CLS token (pre-head) and L2 normalize.
- Retrieval accuracy: for query i (view1), does nearest key in view2 match i?

Outputs:
- Prints ok=true, passed=true|false and metrics.
- Writes a JSON metrics file next to the checkpoint by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is importable when running as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from scripts.phase5_big_run import DinoStudentTeacher, PatchViT


class TwoCrops:
    def __init__(self, t1: transforms.Compose, t2: transforms.Compose) -> None:
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img):
        return [self.t1(img), self.t2(img)]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cifar_views(img_size: int) -> TwoCrops:
    # Match scripts/baseline_cifar10_pretrain.py.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    base = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    t1 = transforms.Compose(base)
    t2 = transforms.Compose(base)
    return TwoCrops(t1, t2)


@torch.no_grad()
def _embed_backbone_cls(student: DinoStudentTeacher, x: torch.Tensor) -> torch.Tensor:
    feats = student.backbone(x)  # (B, N+1, D)
    cls = feats[:, 0]
    return F.normalize(cls.float(), p=2, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="CIFAR-10 label-free view-retrieval eval")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/_torchvision"))
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=10.0, help="Pass gate: top1 >= ratio*(1/N)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: next to checkpoint (cifar_view_retrieval_step<step>_split<split>_N<n>.json)",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.topk <= 0:
        raise SystemExit("--topk must be > 0")

    _seed_all(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(payload.get("step", 0) or 0)

    cfg = payload.get("config") or {}
    model = cfg.get("model") or {}

    img_size = int(model.get("img_size", 32))
    patch = int(model.get("patch", 4))
    dim = int(model.get("dim", 384))
    depth = int(model.get("depth", 6))
    heads = int(model.get("heads", 6))
    mlp_ratio = float(model.get("mlp_ratio", 4.0))
    out_dim = int(model.get("out_dim", 8192))

    vit = PatchViT(
        img_size=img_size,
        patch=patch,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        use_grad_checkpoint=False,
    )
    student = DinoStudentTeacher(vit, out_dim=out_dim)
    student.load_state_dict(payload["student"], strict=True)
    student.to(device)
    student.eval()

    views = _cifar_views(img_size)
    ds = datasets.CIFAR10(
        root=str(args.data_root),
        train=(args.split == "train"),
        download=True,
        transform=views,
    )

    n = int(args.n)
    if n > len(ds):
        n = len(ds)

    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), k=n)

    t0 = time.time()
    Q_chunks: list[torch.Tensor] = []
    K_chunks: list[torch.Tensor] = []

    bs = int(args.batch_size)
    for start in range(0, n, bs):
        end = min(n, start + bs)

        v1_list = []
        v2_list = []
        for j in range(start, end):
            (v1, v2), _y = ds[idxs[j]]
            v1_list.append(v1)
            v2_list.append(v2)

        x1 = torch.stack(v1_list, dim=0).to(device, non_blocking=True)
        x2 = torch.stack(v2_list, dim=0).to(device, non_blocking=True)

        q = _embed_backbone_cls(student, x1)
        k = _embed_backbone_cls(student, x2)

        Q_chunks.append(q.cpu())
        K_chunks.append(k.cpu())

    Q = torch.cat(Q_chunks, dim=0)
    K = torch.cat(K_chunks, dim=0)

    S = (Q.float() @ K.float().T).numpy()  # (N,N)

    top1_idx = np.argmax(S, axis=1)
    top1 = float(np.mean(top1_idx == np.arange(n)))

    k = int(args.topk)
    if k >= n:
        k = n
    topk_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    topk = float(np.mean([(i in topk_idx[i]) for i in range(n)]))

    baseline = 1.0 / float(n)
    ratio = top1 / baseline if baseline > 0 else float("inf")
    passed = top1 >= float(args.ratio) * baseline

    dt = time.time() - t0

    out = args.out
    if out is None:
        out = args.checkpoint.parent / f"cifar_view_retrieval_step{step}_split{args.split}_N{n}.json"

    metrics = {
        "kind": "cifar10_view_retrieval",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": step,
        "split": args.split,
        "n": n,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "topk": int(args.topk),
        "top1": top1,
        "topk_acc": topk,
        "random_baseline": baseline,
        "ratio_vs_random": ratio,
        "pass_ratio": float(args.ratio),
        "passed": bool(passed),
        "seconds": float(dt),
        "model": {
            "img_size": img_size,
            "patch": patch,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
            "out_dim": out_dim,
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

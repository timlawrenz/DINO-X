#!/usr/bin/env python3
"""Baseline: CIFAR-10 linear probe evaluation.

Loads a CIFAR-10 DINO pretrain checkpoint and trains a frozen-backbone linear
classifier. Outputs an objective top-1 accuracy and pass/fail signal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Ensure repository root is importable when running as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from scripts.phase5_big_run import DinoStudentTeacher, PatchViT


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def main() -> int:
    ap = argparse.ArgumentParser(description="CIFAR-10 linear probe")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/_torchvision"))

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    args = ap.parse_args()

    _seed_all(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    model = (cfg.get("model") or {})

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

    backbone = student.backbone
    for p in backbone.parameters():
        p.requires_grad_(False)

    clf = nn.Linear(dim, 10).to(device)

    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.CIFAR10(root=str(args.data_root), train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root=str(args.data_root), train=False, download=True, transform=tfm)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    opt = torch.optim.SGD(clf.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()

    for epoch in range(args.epochs):
        clf.train()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                feats = backbone(x)
                cls = feats[:, 0]

            logits = clf(cls)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Eval
        clf.eval()
        accs: list[float] = []
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = backbone(x)
                cls = feats[:, 0]
                logits = clf(cls)
                accs.append(_accuracy_top1(logits, y))

        acc = float(np.mean(accs) if accs else 0.0)
        print(f"epoch={epoch+1:03d}/{args.epochs} test_top1={acc:.4f}")

    elapsed = time.time() - t0

    passed = acc >= float(args.threshold)

    out_dir = args.checkpoint.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"linear_probe_{ts}.json"

    metrics = {
        "kind": "cifar10_linear_probe",
        "version": 1,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "test_top1": float(acc),
        "threshold": float(args.threshold),
        "passed": bool(passed),
        "elapsed_s": float(elapsed),
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print("ok=true")
    print(f"metrics_json={out_path}")
    print(f"passed={str(passed).lower()}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

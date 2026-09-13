"""Ablation: does true 3-slice (2.5D) context beat single-slice duplication?

The shipped model's released inference path (predict_proba) and its external
validation (eval_external.py) BOTH feed a single slice duplicated across the 3
input channels: [z, z, z]. The pretrained backbone was trained with true adjacent
context [z-1, z, z+1]. An external reviewer asked: how much slice-level accuracy is
left on the table by discarding spatial context at inference?

This script recomputes the external CRLM slice AUROC two ways on the SAME model
(no retraining, patch_embed is a frozen Conv2d(3->dim) untouched by LoRA):
  A) duplicated single slice  [z, z, z]      (the shipped/validated path)
  B) true adjacent context    [z-1, z, z+1]  (boundary slices clamp to edge)

Verdict is empirical: compare slice AUROC (and accuracy@0.5) A vs B.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoo.hub import load_model
from zoo.peft import load_adapter

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _idx(p: str) -> int:
    return int(re.search(r"slice_(\d+)", p).group(1))


class CtxDataset(Dataset):
    """Loads (z-1, z, z+1) for each labeled slice; clamps at volume boundaries."""

    def __init__(self, rows, img_size, level, width, mode):
        assert mode in ("dup", "ctx")
        self.rows = rows
        self.mode = mode
        self.level, self.width = level, width
        # group slice indices present per patient dir for neighbor lookup
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.rows)

    def _load_windowed(self, path: Path) -> np.ndarray:
        arr = np.array(Image.open(path), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        wmin = self.level - self.width / 2.0
        return np.clip((hu - wmin) / max(self.width, 1.0), 0.0, 1.0)

    def _neighbor(self, path: Path, delta: int) -> Path:
        i = _idx(str(path))
        cand = path.parent / f"slice_{i + delta:04d}.png"
        return cand if cand.exists() else path  # clamp to self at boundary

    def __getitem__(self, i):
        r = self.rows[i]
        z = Path(r["png_path"])
        z = z if z.is_absolute() else z
        mid = self._load_windowed(z)
        if self.mode == "dup":
            stack = [mid, mid, mid]
        else:
            stack = [self._load_windowed(self._neighbor(z, -1)), mid,
                     self._load_windowed(self._neighbor(z, +1))]
        x = np.stack(stack, axis=0)
        x = self.transform(torch.from_numpy(x).contiguous())
        sp = torch.tensor([float(r["spacing_x"]), float(r["spacing_y"]), float(r["spacing_z"])], dtype=torch.float32)
        return x, sp, float(r["label"]), r["patient_id"]


def load_model_full(adapter_dir: Path, device):
    cfg = json.load(open(adapter_dir / "finetune_config.json"))
    backbone = load_model(cfg["backbone"], device=str(device))
    dim = backbone.dim
    backbone = load_adapter(backbone, adapter_dir)
    head = nn.Linear(dim, cfg["num_classes"]).to(device)
    backbone = backbone.to(device).eval()
    head.load_state_dict(torch.load(adapter_dir / "head.pth", map_location=device))
    head.eval()
    return backbone, head


@torch.no_grad()
def run(backbone, head, loader, device):
    probs, labels, pids = [], [], []
    for x, sp, y, pid in loader:
        x = x.to(device)
        sp = sp.to(device)
        logits = head(backbone(x, spacing=sp)[:, 0])
        p = torch.softmax(logits, -1)[:, 1]
        probs.append(p.cpu().numpy()); labels.append(y.numpy()); pids.extend(pid)
    return np.concatenate(probs), np.concatenate(labels), np.array(pids)


def auroc(probs, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters/msd-hepatic-vessel-vit-base-50k")
    ap.add_argument("--label-csv", default="data/crlm/vessel_labels_local.csv")
    ap.add_argument("--level", type=float, default=-30.0)
    ap.add_argument("--width", type=float, default=120.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap N slices")
    ap.add_argument("--out", default="results/ablation_3slice_context.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(args.label_csv)))
    if args.limit:
        rows = rows[: args.limit]
    print(f"evaluating {len(rows)} slices on {device}")

    backbone, head = load_model_full(Path(args.adapter), device)
    img_size = backbone.dim and args.img_size

    results = {}
    for mode in ("dup", "ctx"):
        ds = CtxDataset(rows, args.img_size, args.level, args.width, mode)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
        probs, labels, pids = run(backbone, head, loader, device)
        acc = float(((probs >= 0.5).astype(int) == labels).mean())
        results[mode] = {"slice_auroc": auroc(probs, labels), "acc@0.5": acc,
                         "n": len(labels), "pos": int(labels.sum())}
        print(f"[{mode}] AUROC={results[mode]['slice_auroc']:.4f} acc@0.5={acc:.4f}", flush=True)

    results["delta_auroc_ctx_minus_dup"] = results["ctx"]["slice_auroc"] - results["dup"]["slice_auroc"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

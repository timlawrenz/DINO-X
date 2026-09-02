#!/usr/bin/env python3
"""External held-out evaluation of a DINO-X LoRA adapter.

Loads a pretrained backbone + LoRA adapter + task head, runs inference on
a label CSV (same schema as finetune_lora.py), and reports BOTH slice-level
and patient-level AUROC/metrics. Patient-level is the decision-relevant unit
(aggregate slices per patient via mean score), per the evaluation-design gates
in the scientific-experiment-structure skill.

Usage:
    python scripts/eval_external.py \
      --adapter adapters/msd-hepatic-vessel-vit-base-50k \
      --label-csv data/crlm/vessel_labels.csv \
      --window-level -30 --window-width 120 \
      --device cuda
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zoo.arch import PatchViT
from zoo.hub import load_model
from zoo.peft import load_adapter


class EvalDataset(Dataset):
    def __init__(self, rows, img_size, window_level, window_width, data_root):
        self.rows = rows
        self.img_size = img_size
        self.window_level = window_level
        self.window_width = window_width
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def _resolve(self, p):
        p = Path(p)
        if p.is_absolute():
            return p
        if self.data_root:
            return self.data_root / p
        return p

    def _load_image(self, p):
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        wmin = self.window_level - self.window_width / 2.0
        windowed = (hu - wmin) / max(self.window_width, 1.0)
        return np.clip(windowed, 0.0, 1.0)

    def __getitem__(self, idx):
        row = self.rows[idx]
        arr = self._load_image(self._resolve(row["png_path"]))
        x = np.stack([arr, arr, arr], axis=0)
        x = self.transform(torch.from_numpy(x).contiguous())
        spacing = torch.tensor([row["spacing_x"], row["spacing_y"], row["spacing_z"]], dtype=torch.float32)
        return x, spacing, float(row["label"]), row["patient_id"]


def load_adapter_model(args, device):
    """Reconstruct the fine-tuned model: backbone + LoRA adapter + head."""
    cfg = json_load(args.adapter / "finetune_config.json")
    backbone = load_model(cfg["backbone"], device=str(device))
    dim = backbone.dim
    # Load the trained LoRA adapter directly. PeftModel.from_pretrained reads
    # rank/alpha/target_modules from adapter_config.json and injects LoRA — do NOT
    # call apply_lora() here or the backbone gets double-wrapped and the adapter
    # weights won't attach (silent random-LoRA inference).
    backbone = load_adapter(backbone, args.adapter)
    head = nn.Linear(dim, cfg["num_classes"]).to(device)
    # Ensure all modules on the target device (PEFT wrap may reset to CPU)
    backbone = backbone.to(device)
    head = head.to(device)
    head.load_state_dict(torch.load(args.adapter / "head.pth", map_location=device))
    return backbone, head, cfg


def json_load(p):
    import json
    with open(p) as f:
        return json.load(f)


@torch.no_grad()
def run_eval(model, head, loader, device, scale_aware):
    model.eval()
    head.eval()
    all_probs, all_labels, all_pids = [], [], []
    n_done = 0
    for images, spacings, labels, pids in loader:
        images = images.to(device)
        spacing = spacings.to(device) if scale_aware else None
        logits = head(model(images, spacing=spacing)[:, 0])  # CLS token (matches FinetuneModel.forward)
        probs = torch.softmax(logits, dim=-1)[:, 1] if logits.shape[-1] == 2 else logits[:, 0]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_pids.extend(pids)
        n_done += images.shape[0]
        if n_done % 2000 < images.shape[0] or n_done == len(loader.dataset):
            print(f"  progress: {n_done}/{len(loader.dataset)} slices", flush=True)
    return (np.concatenate(all_probs), np.concatenate(all_labels), np.array(all_pids))


def auroc(probs, labels):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    # tie handling
    sorted_p = probs[order]
    i = 0
    while i < len(sorted_p):
        j = i + 1
        while j < len(sorted_p) and sorted_p[j] == sorted_p[i]:
            j += 1
        if j > i + 1:
            avg = np.mean(np.arange(i + 1, j + 1))
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def patient_auroc(probs, labels, pids):
    # aggregate per patient: mean of slice probs, then AUROC over patients
    from collections import defaultdict
    agg_p, agg_l = defaultdict(list), defaultdict(int)
    for p, l in zip(pids, labels):
        agg_p[p].append(float(p))
        agg_l[p] = int(l)
    pat_probs = [float(np.mean(agg_p[p])) for p in agg_p]
    pat_labels = [agg_l[p] for p in agg_p]
    return auroc(np.array(pat_probs), np.array(pat_labels)), len(pat_labels)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--label-csv", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--window-level", type=float, default=-30.0)
    ap.add_argument("--window-width", type=float, default=120.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    rows = []
    with open(args.label_csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "png_path": r["png_path"],
                "label": float(r["label"]),
                "spacing_x": float(r["spacing_x"]),
                "spacing_y": float(r["spacing_y"]),
                "spacing_z": float(r["spacing_z"]),
                "patient_id": r["patient_id"],
            })
    print(f"Loaded {len(rows)} labeled slices")
    pos = sum(1 for r in rows if r["label"] == 1)
    print(f"  positive: {pos} ({100*pos/len(rows):.1f}%), patients: {len(set(r['patient_id'] for r in rows))}")

    model, head, cfg = load_adapter_model(args, device)
    scale_aware = cfg["scale_aware"]
    print(f"Backbone scale_aware={scale_aware}, num_classes={cfg['num_classes']}, rank={cfg['rank']}")

    ds = EvalDataset(rows, img_size=model.img_size, window_level=args.window_level,
                     window_width=args.window_width, data_root=args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probs, labels, pids = run_eval(model, head, loader, device, scale_aware)
    slice_auc = auroc(probs, labels)
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    pat_auc, n_pat = patient_auroc(probs, labels, pids)

    print("\n===== EXTERNAL EVAL RESULTS =====")
    print(f"Slice-level AUROC: {slice_auc:.4f}")
    print(f"Slice-level accuracy (thr 0.5): {acc:.4f}")
    print(f"Patient-level AUROC: {pat_auc:.4f}  (n_patients={n_pat})")
    # severity
    if slice_auc >= 0.90:
        sev = "EXCELLENT"
    elif slice_auc >= 0.80:
        sev = "STRONG"
    elif slice_auc >= 0.70:
        sev = "ADEQUATE"
    else:
        sev = "WEAK"
    print(f"Verdict: {sev} external generalization (transfer to new institution's scans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
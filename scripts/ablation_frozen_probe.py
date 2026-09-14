"""Ablation: can the FROZEN backbone (no LoRA) separate vessel vs non-vessel
under the narrow (-30/120) window, on external CRLM data?

The 3rd external review claims a "massive domain shift" between the wide random
pretraining windows (level +/-400, width 800-2000) and the narrow fine-tune window
(-30/120), and that expecting the frozen backbone + a lightweight LoRA to bridge it
is "structurally questionable."

Test: extract CLS features from the frozen scale-aware backbone (NO LoRA adapter)
on external CRLM slices windowed at -30/120, fit a plain logistic-regression probe
on the SCALAR spacing-free features, and measure AUROC. If the frozen features
alone carry the vessel signal, the narrow window is not a structural blocker.
If AUROC ~ 0.5, the reviewer is right that the window breaks the backbone.

We use a held-out patient split (group by patient_id) so the probe number is not
slice-autocorrelation-inflated.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoo.hub import load_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FeatDataset(Dataset):
    def __init__(self, rows, img_size, level, width):
        self.rows = rows
        self.level, self.width = level, width
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = Path(r["png_path"])
        arr = np.array(Image.open(p), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        wmin = self.level - self.width / 2.0
        w = np.clip((hu - wmin) / max(self.width, 1.0), 0.0, 1.0)
        x = np.stack([w, w, w], axis=0)  # duplicated slice = the shipped path
        x = self.transform(torch.from_numpy(x).contiguous())
        sp = torch.tensor([float(r["spacing_x"]), float(r["spacing_y"]), float(r["spacing_z"])])
        return x, sp, float(r["label"]), r["patient_id"]


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="runs/20260728_180949_hepatic-specialist-vit-base-scale-aware/checkpoint_final_00050000.pth")
    ap.add_argument("--label-csv", default="data/crlm/vessel_labels_local.csv")
    ap.add_argument("--level", type=float, default=-30.0)
    ap.add_argument("--width", type=float, default=120.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", default="results/ablation_frozen_probe.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(args.label_csv)))
    if args.limit:
        rows = rows[: args.limit]
    print(f"extracting frozen-backbone features for {len(rows)} slices on {device} (NO LoRA)")

    backbone = load_model(args.backbone, device=device).to(device).eval()
    ds = FeatDataset(rows, args.img_size, args.level, args.width)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    feats, labels, pids = [], [], []
    for x, sp, y, pid in loader:
        x = x.to(device); sp = sp.to(device)
        f = backbone(x, spacing=sp)[:, 0]  # CLS, frozen backbone, NO adapter
        feats.append(f.cpu().numpy()); labels.append(y.numpy()); pids.extend(pid)
    X = np.concatenate(feats); y = np.concatenate(labels); pids = np.array(pids)
    print("features:", X.shape)

    # Patient-grouped split: train probe on 70% patients, test on held-out 30%.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr, te = next(gss.split(X, y, groups=pids))
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    prob = clf.predict_proba(X[te])[:, 1]
    auroc = float(roc_auc_score(y[te], prob))

    # Null: raw-pixel-intensity baseline (does the window alone carry it?)
    intensity = X.mean(axis=1)
    null_auroc = float(roc_auc_score(y[te], intensity[te]))

    out = {
        "frozen_backbone_linear_probe_auroc": auroc,
        "null_intensity_probe_auroc": null_auroc,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_train_patients": int(len(set(pids[tr]))), "n_test_patients": int(len(set(pids[te]))),
        "note": "Frozen backbone CLS features (no LoRA), -30/120 window, patient-grouped split",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

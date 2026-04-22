#!/usr/bin/env python3
"""Pan-organ evaluation protocol for DINO-X models.

Implements 6 metrics to evaluate self-supervised CT representations:

Primary metrics:
  1. Per-dataset view retrieval — sanity check (existing protocol, per dataset)
  2. Dataset discrimination probe — LogisticRegression on CLS features (series-split)
  3. Spacing counterfactual test — same pixels, varied spacing → embedding delta

Secondary metrics:
  4. Domain clustering analysis — NN composition adjusted for prevalence
  5. Spacing prediction sanity check — Ridge regression on CLS features
  6. Embedding statistics — per-dataset StdDev, cross-dataset cosine

Design decisions:
  - Deterministic eval transforms (center crop, fixed HU window, no augmentation)
  - Series-level train/test splits for probes (no slice leakage)
  - Backbone CLS token used for all probes (not projection head)
  - Spacing passed to backbone when --scale-aware
  - Bootstrap CIs computed at series level

Usage:
    python scripts/evaluate_panorgan.py \\
        --checkpoint path/to/checkpoint.pth \\
        --index-csv data/processed/combined-mvp/index.csv \\
        --split-manifest data/processed/combined-mvp/split_manifest.json \\
        --scale-aware \\
        --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. Install it and retry.")


try:
    import numpy as np
except Exception:
    _need("numpy")

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except Exception:
    _need("torch")

try:
    from PIL import Image
except Exception:
    _need("pillow")


# ─────────────────────────────────────────────────────────────────────────────
# Add repo root and scripts/ to path for local imports
_scripts_dir = str(Path(__file__).resolve().parent)
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from zoo.arch import (  # noqa: E402
    DinoStudentTeacher,
    PatchViT,
    migrate_state_dict,
    needs_migration,
)
from phase5_big_run import ModelConfig, PngDataset, _load_index_rows  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic eval dataset (no random augmentation)
# ─────────────────────────────────────────────────────────────────────────────

class EvalDataset(torch.utils.data.Dataset):
    """Deterministic eval dataset with fixed HU window and center crop.

    Unlike PngDataset (random crop/flip/window), this uses:
    - Fixed soft-tissue window (L=40, W=400) for consistent evaluation
    - Center crop to model input size
    - No horizontal flip
    - Same ImageNet normalization as training
    """

    def __init__(
        self,
        rows: list,  # list[IndexRow]
        img_size: int = 224,
        window_level: float = 40.0,
        window_width: float = 400.0,
    ):
        self.rows = rows
        self.img_size = img_size
        self.window_level = window_level
        self.window_width = window_width

        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Map series for 3-slice context
        self._series_map: dict[str, dict[int, Path]] = {}
        self._series_minmax: dict[str, tuple[int, int]] = {}
        for r in rows:
            sm = self._series_map.setdefault(r.series_dir, {})
            sm[r.slice_index] = r.png_path
        for s, mp in self._series_map.items():
            if mp:
                ks = sorted(mp.keys())
                self._series_minmax[s] = (ks[0], ks[-1])

    def __len__(self) -> int:
        return len(self.rows)

    def _load_hu01(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        hu = (arr - 32768.0) * 0.1
        wmin = self.window_level - self.window_width / 2.0
        wmax = self.window_level + self.window_width / 2.0
        windowed = (hu - wmin) / max(self.window_width, 1.0)
        return np.clip(windowed, 0.0, 1.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        s = row.series_dir
        z = row.slice_index
        z0, z1 = self._series_minmax.get(s, (z, z))

        def _clamp(k: int) -> int:
            return max(z0, min(z1, k))

        mp = self._series_map.get(s, {})
        paths = [
            mp.get(_clamp(z - 1), row.png_path),
            mp.get(_clamp(z), row.png_path),
            mp.get(_clamp(z + 1), row.png_path),
        ]

        slices = [self._load_hu01(p) for p in paths]
        x = np.stack(slices, axis=0)
        x = self.transform(torch.from_numpy(x).contiguous())

        spacing = torch.tensor(
            [row.spacing_x, row.spacing_y, row.spacing_z],
            dtype=torch.float32,
        )
        return x, spacing


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_backbone_cls(
    student: Any,
    x: torch.Tensor,
    spacing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return L2-normalized CLS embedding from backbone (not projection head)."""
    feats = student.backbone(x, spacing=spacing)
    cls = feats[:, 0]
    return F.normalize(cls.float(), p=2, dim=-1)


@torch.no_grad()
def embed_all(
    student: Any,
    rows: list,
    img_size: int,
    device: torch.device,
    batch_size: int = 64,
    scale_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed all rows deterministically, returning (embeddings, spacings) as numpy.

    Returns:
        embeddings: (N, D) float32 — L2-normalized CLS tokens
        spacings: (N, 3) float32 — (spacing_x, spacing_y, spacing_z)
    """
    ds = EvalDataset(rows, img_size=img_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    emb_chunks = []
    sp_chunks = []

    for imgs, sp in loader:
        imgs = imgs.to(device, non_blocking=True)
        spacing_in = sp.to(device, non_blocking=True) if scale_aware else None
        emb = embed_backbone_cls(student, imgs, spacing=spacing_in)
        emb_chunks.append(emb.cpu().numpy())
        sp_chunks.append(sp.numpy())

    return np.concatenate(emb_chunks, axis=0), np.concatenate(sp_chunks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Per-dataset view retrieval
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def metric_view_retrieval_per_dataset(
    student: Any,
    rows: list,
    img_size: int,
    cfg: dict,
    device: torch.device,
    scale_aware: bool,
    n_per_dataset: int = 512,
    seed: int = 42,
    topk: int = 5,
) -> dict:
    """Run view retrieval independently per dataset (uses random augmentation)."""

    datasets: dict[str, list] = defaultdict(list)
    for r in rows:
        datasets[r.dataset or "unknown"].append(r)

    results = {}
    for ds_name, ds_rows in sorted(datasets.items()):
        rng = random.Random(seed)
        n = min(n_per_dataset, len(ds_rows))
        idxs = rng.sample(range(len(ds_rows)), k=n)

        ds = PngDataset(
            ds_rows,
            img_size=img_size,
            rw_level_min=float(cfg.get("rw_level_min", -400.0)),
            rw_level_max=float(cfg.get("rw_level_max", 400.0)),
            rw_width_min=float(cfg.get("rw_width_min", 800.0)),
            rw_width_max=float(cfg.get("rw_width_max", 2000.0)),
        )

        Q_chunks, K_chunks = [], []
        bs = 64
        for start in range(0, n, bs):
            end = min(n, start + bs)
            v1_list, v2_list = [], []
            for j in range(start, end):
                views, _sp = ds[idxs[j]]
                v1_list.append(views[0])
                v2_list.append(views[1])

            x1 = torch.stack(v1_list).to(device)
            x2 = torch.stack(v2_list).to(device)

            sp_batch = None
            if scale_aware:
                sp_batch = torch.stack([
                    torch.tensor([ds_rows[idxs[j]].spacing_x,
                                  ds_rows[idxs[j]].spacing_y,
                                  ds_rows[idxs[j]].spacing_z])
                    for j in range(start, end)
                ]).to(device)

            Q_chunks.append(embed_backbone_cls(student, x1, sp_batch).cpu())
            K_chunks.append(embed_backbone_cls(student, x2, sp_batch).cpu())

        Q = torch.cat(Q_chunks).numpy()
        K = torch.cat(K_chunks).numpy()

        S = Q @ K.T
        top1_idx = np.argmax(S, axis=1)
        top1 = float(np.mean(top1_idx == np.arange(n)))
        baseline = 1.0 / n
        ratio = top1 / baseline if baseline > 0 else float("inf")

        topk_idx = np.argpartition(-S, kth=min(topk, n) - 1, axis=1)[:, :topk]
        topk_acc = float(np.mean([i in topk_idx[i] for i in range(n)]))

        results[ds_name] = {
            "n": n,
            "top1": top1,
            f"top{topk}": topk_acc,
            "random_baseline": baseline,
            "ratio_vs_random": ratio,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Dataset discrimination linear probe
# ─────────────────────────────────────────────────────────────────────────────

def metric_dataset_discrimination_probe(
    embeddings: np.ndarray,
    rows: list,
    seed: int = 42,
) -> dict:
    """Train LogisticRegression on CLS features to classify dataset.

    Split is at series level to prevent slice leakage.
    Reports accuracy and AUC with series-level bootstrap CIs.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        return {"error": "scikit-learn not installed"}

    # Group by series
    series_to_dataset: dict[str, str] = {}
    series_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        series_to_dataset[r.series_dir] = r.dataset or "unknown"
        series_to_indices[r.series_dir].append(i)

    # Stratified series-level split: ensure each dataset has series in both splits
    ds_series: dict[str, list[str]] = defaultdict(list)
    for s, d in series_to_dataset.items():
        ds_series[d].append(s)

    rng = random.Random(seed)
    train_series: set[str] = set()
    test_series: set[str] = set()

    for d in sorted(ds_series.keys()):
        s_list = sorted(ds_series[d])
        rng.shuffle(s_list)
        n_train = max(1, int(0.8 * len(s_list)))
        # Ensure at least 1 in test
        if n_train == len(s_list):
            n_train = max(1, len(s_list) - 1)
        train_series.update(s_list[:n_train])
        test_series.update(s_list[n_train:])

    train_idx = [i for s in train_series for i in series_to_indices[s]]
    test_idx = [i for s in test_series for i in series_to_indices[s]]

    if not train_idx or not test_idx:
        return {"error": "insufficient series for train/test split"}

    # Get unique labels
    all_labels = sorted(set(series_to_dataset.values()))
    label_map = {l: i for i, l in enumerate(all_labels)}

    y_train = np.array([label_map[series_to_dataset[rows[i].series_dir]] for i in train_idx])
    y_test = np.array([label_map[series_to_dataset[rows[i].series_dir]] for i in test_idx])

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    # Check we have at least 2 classes in both splits
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"error": "need at least 2 datasets in both train and test splits"}

    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))

    # AUC: binary if 2 classes, macro-average if >2
    if len(all_labels) == 2:
        auc = float(roc_auc_score(y_test, y_prob[:, 1]))
    else:
        auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))

    # Series-level bootstrap CI for accuracy
    boot_accs = []
    rng_boot = random.Random(seed + 1)
    test_series_list = sorted(test_series)
    for _ in range(200):
        boot_series = [test_series_list[rng_boot.randint(0, len(test_series_list) - 1)]
                       for _ in range(len(test_series_list))]
        boot_idx = [i for s in boot_series for i in series_to_indices[s]]
        if not boot_idx:
            continue
        y_b = np.array([label_map[series_to_dataset[rows[i].series_dir]] for i in boot_idx])
        pred_b = clf.predict(embeddings[boot_idx])
        boot_accs.append(float(accuracy_score(y_b, pred_b)))

    ci_lo = float(np.percentile(boot_accs, 2.5)) if boot_accs else acc
    ci_hi = float(np.percentile(boot_accs, 97.5)) if boot_accs else acc

    return {
        "labels": all_labels,
        "train_series": len(train_series),
        "test_series": len(test_series),
        "train_slices": len(train_idx),
        "test_slices": len(test_idx),
        "accuracy": acc,
        "accuracy_ci95": [ci_lo, ci_hi],
        "auc": auc,
        "note": "dataset discrimination (not organ — confounded by scanner/protocol)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Spacing counterfactual test
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def metric_spacing_counterfactual(
    student: Any,
    rows: list,
    img_size: int,
    device: torch.device,
    n: int = 256,
    seed: int = 42,
) -> dict:
    """Measure embedding sensitivity to spacing metadata (same pixels, different spacing).

    For each sample:
    - Embed with real spacing → e_real
    - Embed with 2× spacing  → e_2x
    - Embed with 0.5× spacing → e_half
    - Measure cosine distance between embeddings

    Expected behavior:
    - Baseline (no scale embed): distances ≈ 0 (ignores spacing)
    - Scale-aware: distances > 0, smooth and proportional
    """
    rng = random.Random(seed)
    sample_idx = rng.sample(range(len(rows)), k=min(n, len(rows)))
    sample_rows = [rows[i] for i in sample_idx]

    ds = EvalDataset(sample_rows, img_size=img_size)

    dist_real_2x = []
    dist_real_half = []
    dist_half_2x = []

    bs = 64
    for start in range(0, len(sample_rows), bs):
        end = min(len(sample_rows), start + bs)
        imgs_list = []
        sp_real_list = []

        for j in range(start, end):
            img, sp = ds[j]
            imgs_list.append(img)
            sp_real_list.append(sp)

        imgs = torch.stack(imgs_list).to(device)
        sp_real = torch.stack(sp_real_list).to(device)
        sp_2x = sp_real * 2.0
        sp_half = sp_real * 0.5

        e_real = embed_backbone_cls(student, imgs, spacing=sp_real)
        e_2x = embed_backbone_cls(student, imgs, spacing=sp_2x)
        e_half = embed_backbone_cls(student, imgs, spacing=sp_half)

        # Cosine distance = 1 - cosine_similarity
        dist_real_2x.extend((1.0 - (e_real * e_2x).sum(dim=-1)).cpu().tolist())
        dist_real_half.extend((1.0 - (e_real * e_half).sum(dim=-1)).cpu().tolist())
        dist_half_2x.extend((1.0 - (e_half * e_2x).sum(dim=-1)).cpu().tolist())

    return {
        "n": len(sample_rows),
        "cosine_distance_real_vs_2x": {
            "mean": float(np.mean(dist_real_2x)),
            "std": float(np.std(dist_real_2x)),
            "median": float(np.median(dist_real_2x)),
        },
        "cosine_distance_real_vs_half": {
            "mean": float(np.mean(dist_real_half)),
            "std": float(np.std(dist_real_half)),
            "median": float(np.median(dist_real_half)),
        },
        "cosine_distance_half_vs_2x": {
            "mean": float(np.mean(dist_half_2x)),
            "std": float(np.std(dist_half_2x)),
            "median": float(np.median(dist_half_2x)),
        },
        "interpretation": (
            "Baseline: distances ~0 (model ignores spacing metadata). "
            "Scale-aware: distances > 0 (model encodes physical scale)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Domain clustering analysis
# ─────────────────────────────────────────────────────────────────────────────

def metric_domain_clustering(
    embeddings: np.ndarray,
    rows: list,
    k: int = 10,
) -> dict:
    """Analyze nearest neighbor composition — measures domain clustering.

    For each embedding, find k nearest neighbors and check what fraction
    come from the same dataset. Adjusted for class prevalence.
    """
    datasets = [r.dataset or "unknown" for r in rows]
    unique_ds = sorted(set(datasets))
    ds_idx = {d: i for i, d in enumerate(unique_ds)}
    labels = np.array([ds_idx[d] for d in datasets])

    # Prevalence (expected same-dataset rate under random retrieval)
    prevalence = {d: float(np.mean(labels == ds_idx[d])) for d in unique_ds}

    # Cosine similarity matrix
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, -float("inf"))  # exclude self

    topk_idx = np.argpartition(-S, kth=k, axis=1)[:, :k]

    # Same-dataset fraction per sample
    same_frac = []
    for i in range(len(rows)):
        neighbors = topk_idx[i]
        same_count = int(np.sum(labels[neighbors] == labels[i]))
        same_frac.append(same_count / k)

    # Per-dataset stats
    per_ds = {}
    for d in unique_ds:
        mask = labels == ds_idx[d]
        fracs = [same_frac[i] for i in range(len(rows)) if mask[i]]
        expected = prevalence[d]
        observed = float(np.mean(fracs))
        per_ds[d] = {
            "same_dataset_rate": observed,
            "expected_random": expected,
            "enrichment": observed / expected if expected > 0 else float("inf"),
            "n": int(np.sum(mask)),
        }

    overall_same = float(np.mean(same_frac))
    expected_overall = sum(prevalence[d] ** 2 for d in unique_ds)

    return {
        "k": k,
        "overall_same_dataset_rate": overall_same,
        "expected_random_rate": expected_overall,
        "enrichment_vs_random": overall_same / expected_overall if expected_overall > 0 else float("inf"),
        "per_dataset": per_ds,
        "note": "High enrichment = strong domain clustering. Not necessarily good or bad.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 5: Spacing prediction sanity check
# ─────────────────────────────────────────────────────────────────────────────

def metric_spacing_prediction(
    embeddings: np.ndarray,
    spacings: np.ndarray,
    rows: list,
    seed: int = 42,
) -> dict:
    """Ridge regression from CLS features to log(spacing_x).

    Sanity check: if scale-aware model encodes spacing, R² should be high.
    Note: this is partly circular for scale-aware models (spacing is added to tokens).
    Series-level split to prevent leakage.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
    except ImportError:
        return {"error": "scikit-learn not installed"}

    # Stratified series-level split (ensure spacing variance in both splits)
    series_to_indices: dict[str, list[int]] = defaultdict(list)
    series_to_dataset: dict[str, str] = {}
    for i, r in enumerate(rows):
        series_to_indices[r.series_dir].append(i)
        series_to_dataset[r.series_dir] = r.dataset or "unknown"

    ds_series: dict[str, list[str]] = defaultdict(list)
    for s, d in series_to_dataset.items():
        ds_series[d].append(s)

    rng = random.Random(seed)
    train_series_set: set[str] = set()
    test_series_set: set[str] = set()

    for d in sorted(ds_series.keys()):
        s_list = sorted(ds_series[d])
        rng.shuffle(s_list)
        n_train = max(1, int(0.8 * len(s_list)))
        if n_train == len(s_list):
            n_train = max(1, len(s_list) - 1)
        train_series_set.update(s_list[:n_train])
        test_series_set.update(s_list[n_train:])

    train_idx = [i for s in train_series_set for i in series_to_indices[s]]
    test_idx = [i for s in test_series_set for i in series_to_indices[s]]

    if not train_idx or not test_idx:
        return {"error": "insufficient series for split"}

    # Target: log(spacing_x) — log scale for numerical stability
    y = np.log(spacings[:, 0] + 1e-6)

    X_train, y_train = embeddings[train_idx], y[train_idx]
    X_test, y_test = embeddings[test_idx], y[test_idx]

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    return {
        "target": "log(spacing_x)",
        "train_slices": len(train_idx),
        "test_slices": len(test_idx),
        "r2": r2,
        "mae_log_spacing": mae,
        "note": "Partly circular for scale-aware models. Use as plumbing check.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 6: Embedding statistics
# ─────────────────────────────────────────────────────────────────────────────

def metric_embedding_stats(
    embeddings: np.ndarray,
    spacings: np.ndarray,
    rows: list,
) -> dict:
    """Embedding diversity and structure statistics."""
    datasets = [r.dataset or "unknown" for r in rows]
    unique_ds = sorted(set(datasets))

    per_ds = {}
    ds_centroids = {}
    for d in unique_ds:
        mask = np.array([ds == d for ds in datasets])
        emb_d = embeddings[mask]
        sp_d = spacings[mask]

        centroid = emb_d.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        ds_centroids[d] = centroid

        # Intra-dataset cosine similarity
        intra_cos = float((emb_d @ centroid).mean())

        # Embedding StdDev (diversity)
        emb_std = float(emb_d.std(axis=0).mean())

        # PCA-spacing correlation (first PC vs spacing_x)
        if emb_d.shape[0] > 2:
            emb_centered = emb_d - emb_d.mean(axis=0)
            _, _, Vt = np.linalg.svd(emb_centered, full_matrices=False)
            pc1 = emb_centered @ Vt[0]
            corr = float(np.corrcoef(pc1, sp_d[:, 0])[0, 1])
        else:
            corr = float("nan")

        per_ds[d] = {
            "n": int(mask.sum()),
            "embedding_std": emb_std,
            "intra_cosine_to_centroid": intra_cos,
            "pca1_spacing_correlation": corr,
        }

    # Cross-dataset centroid similarity
    cross = {}
    ds_names = sorted(ds_centroids.keys())
    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            cos = float(ds_centroids[ds_names[i]] @ ds_centroids[ds_names[j]])
            cross[f"{ds_names[i]}_vs_{ds_names[j]}"] = cos

    return {
        "per_dataset": per_ds,
        "cross_dataset_centroid_cosine": cross,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Pan-organ evaluation protocol for DINO-X")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/combined-mvp/index.csv"))
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--scale-aware", action="store_true", help="Enable scale embedding")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-retrieval", type=int, default=512, help="Samples per dataset for view retrieval")
    ap.add_argument("--n-counterfactual", type=int, default=256, help="Samples for spacing counterfactual")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--skip-view-retrieval", action="store_true", help="Skip view retrieval (requires PNGs)")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.index_csv.exists():
        raise FileNotFoundError(f"index_csv not found: {args.index_csv}")
    if not args.split_manifest.exists():
        raise FileNotFoundError(f"split_manifest not found: {args.split_manifest}")

    # Deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Migrate old-format state dict keys if needed
    if "student" in payload and needs_migration(payload["student"]):
        warnings.warn("Migrating old-format student state dict keys to timm-style")
        payload["student"] = migrate_state_dict(payload["student"])

    step = int(payload.get("step", 0) or 0)
    cfg = payload.get("config", {})
    model_cfg = cfg.get("model", {})

    if isinstance(model_cfg, ModelConfig):
        mc = model_cfg
    else:
        mc = ModelConfig(**model_cfg)

    img_size = int(cfg.get("img_size", 224))

    vit = PatchViT(
        img_size=img_size,
        patch=mc.patch,
        dim=mc.dim,
        depth=mc.depth,
        heads=mc.heads,
        mlp_ratio=mc.mlp_ratio,
        use_grad_checkpoint=False,
        scale_aware=args.scale_aware,
    )
    student = DinoStudentTeacher(vit, out_dim=mc.out_dim).to(device)
    student.load_state_dict(payload["student"], strict=True)
    student.eval()

    # Load val rows
    split_data = json.loads(args.split_manifest.read_text())
    val_series = set(str(s) for s in split_data.get("val", {}).get("series_dir", []))
    all_rows = _load_index_rows(args.index_csv)
    rows = [r for r in all_rows if str(r.series_dir) in val_series]

    if not rows:
        raise SystemExit("No rows after filtering to val series")

    datasets_found = sorted(set(r.dataset for r in rows if r.dataset))
    n_per_ds = {d: sum(1 for r in rows if r.dataset == d) for d in datasets_found}
    print(f"Val set: {len(rows)} slices across {len(datasets_found)} datasets")
    for d, n in n_per_ds.items():
        print(f"  {d}: {n} slices")

    t0 = time.time()
    results: dict[str, Any] = {
        "kind": "panorgan_evaluation",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": step,
        "scale_aware": args.scale_aware,
        "seed": args.seed,
        "val_slices": len(rows),
        "datasets": datasets_found,
        "model": {
            "name": mc.name,
            "patch": mc.patch,
            "dim": mc.dim,
            "depth": mc.depth,
            "heads": mc.heads,
        },
        "metrics": {},
    }

    # ── Metric 1: Per-dataset view retrieval ──
    if not args.skip_view_retrieval:
        print("\n[1/6] Per-dataset view retrieval...")
        try:
            vr = metric_view_retrieval_per_dataset(
                student, rows, img_size, cfg, device,
                scale_aware=args.scale_aware,
                n_per_dataset=args.n_retrieval,
                seed=args.seed,
            )
            results["metrics"]["view_retrieval_per_dataset"] = vr
            for ds_name, ds_res in vr.items():
                print(f"  {ds_name}: top1={ds_res['top1']:.4f} ratio={ds_res['ratio_vs_random']:.1f}×")
        except Exception as e:
            print(f"  ⚠️  Skipped: {e}")
            results["metrics"]["view_retrieval_per_dataset"] = {"error": str(e)}
    else:
        print("\n[1/6] Per-dataset view retrieval... SKIPPED (--skip-view-retrieval)")

    # ── Embed all val slices (deterministic, for metrics 2-6) ──
    print("\n[embed] Embedding all val slices (deterministic)...")
    embeddings, spacings = embed_all(
        student, rows, img_size, device,
        batch_size=args.batch_size,
        scale_aware=args.scale_aware,
    )
    print(f"  Embedded {embeddings.shape[0]} slices → ({embeddings.shape[1]}D)")

    # ── Metric 2: Dataset discrimination probe ──
    print("\n[2/6] Dataset discrimination linear probe...")
    probe = metric_dataset_discrimination_probe(embeddings, rows, seed=args.seed)
    results["metrics"]["dataset_discrimination_probe"] = probe
    if "accuracy" in probe:
        print(f"  Accuracy: {probe['accuracy']:.3f} (CI: {probe['accuracy_ci95']})")
        print(f"  AUC: {probe['auc']:.3f}")
    else:
        print(f"  ⚠️  {probe.get('error', 'unknown error')}")

    # ── Metric 3: Spacing counterfactual ──
    print("\n[3/6] Spacing counterfactual test...")
    if args.scale_aware:
        try:
            cf = metric_spacing_counterfactual(
                student, rows, img_size, device,
                n=args.n_counterfactual, seed=args.seed,
            )
            results["metrics"]["spacing_counterfactual"] = cf
            print(f"  real→2x: dist={cf['cosine_distance_real_vs_2x']['mean']:.4f}")
            print(f"  real→½x: dist={cf['cosine_distance_real_vs_half']['mean']:.4f}")
        except Exception as e:
            print(f"  ⚠️  Skipped: {e}")
            results["metrics"]["spacing_counterfactual"] = {"error": str(e)}
    else:
        print("  Skipped (baseline model has no scale embedding)")
        results["metrics"]["spacing_counterfactual"] = {
            "skipped": True,
            "reason": "baseline model has no scale embedding",
        }

    # ── Metric 4: Domain clustering ──
    print("\n[4/6] Domain clustering analysis...")
    clustering = metric_domain_clustering(embeddings, rows, k=10)
    results["metrics"]["domain_clustering"] = clustering
    print(f"  Same-dataset NN rate: {clustering['overall_same_dataset_rate']:.3f}")
    print(f"  Expected random: {clustering['expected_random_rate']:.3f}")
    print(f"  Enrichment: {clustering['enrichment_vs_random']:.1f}×")

    # ── Metric 5: Spacing prediction ──
    print("\n[5/6] Spacing prediction sanity check...")
    sp_pred = metric_spacing_prediction(embeddings, spacings, rows, seed=args.seed)
    results["metrics"]["spacing_prediction"] = sp_pred
    if "r2" in sp_pred:
        print(f"  R²: {sp_pred['r2']:.3f}")
        print(f"  MAE(log spacing): {sp_pred['mae_log_spacing']:.4f}")
    else:
        print(f"  ⚠️  {sp_pred.get('error', 'unknown error')}")

    # ── Metric 6: Embedding statistics ──
    print("\n[6/6] Embedding statistics...")
    stats = metric_embedding_stats(embeddings, spacings, rows)
    results["metrics"]["embedding_stats"] = stats
    for ds_name, ds_stats in stats["per_dataset"].items():
        print(f"  {ds_name}: std={ds_stats['embedding_std']:.4f} "
              f"intra_cos={ds_stats['intra_cosine_to_centroid']:.3f} "
              f"pca1_sp_corr={ds_stats['pca1_spacing_correlation']:.3f}")
    for pair, cos in stats["cross_dataset_centroid_cosine"].items():
        print(f"  Cross: {pair} = {cos:.3f}")

    # ── Save results ──
    elapsed = time.time() - t0
    results["seconds"] = elapsed

    out = args.out
    if out is None:
        out = args.checkpoint.parent / f"panorgan_eval_step{step}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'─' * 60}")
    print(f"Evaluation complete in {elapsed:.1f}s")
    print(f"Results: {out}")
    print(f"ok=true")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

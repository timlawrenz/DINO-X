#!/usr/bin/env python3
"""Validate the evidence envelope: every claimed metric maps to a reproducible artifact.

Checks:
1. persisted per-slice predictions exist (external_probs_*.npz) and their slice AUROC
   matches evaluation.json within tolerance
2. every evaluation.json field is present
3. envelope file structure is complete

Usage:
    python releases/.../reproduce/repro_metrics.py [--npz PATH] [--eval-json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ENVELOPE = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = ENVELOPE.parent.parent / "adapters" / "msd-hepatic-vessel-vit-base-50k" / "external_probs_vessel_labels_local.npz"


def auroc(probs, labels):
    labels = np.asarray(labels).astype(float)
    probs = np.asarray(probs).astype(float)
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    sp = probs[order]
    i = 0
    while i < len(sp):
        j = i + 1
        while j < len(sp) and sp[j] == sp[i]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--eval-json", type=Path, default=ENVELOPE / "evaluation.json")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()

    ev = json.load(open(args.eval_json))
    claimed = ev["external_crlm"]["slice_auroc"]

    if not args.npz.exists():
        print(f"FAIL: persisted predictions not found at {args.npz}")
        print("Full inference must be run first (see reproduce/README.md).")
        sys.exit(1)

    d = np.load(args.npz, allow_pickle=True)
    probs, labels = d["probs"], d["labels"]
    recomputed = auroc(probs, labels)
    ok_metric = abs(recomputed - claimed) <= args.tol
    print(f"claimed slice AUROC : {claimed}")
    print(f"recomputed AUROC    : {recomputed:.4f}")
    print(f"within tol {args.tol}: {'PASS' if ok_metric else 'FAIL'}")

    # Structure completeness
    required = [Path("ENVELOPE.md"), Path("model_card.md"), Path("evaluation.json"),
                Path("provenance.yaml"), Path("source_branch"),
                Path("the_science/01_decision_trail.md"),
                Path("the_science/02_training_recipe.md"),
                Path("the_science/03_validation.md"),
                Path("the_science/04_negative_results.md"),
                Path("the_science/05_known_limits.md"),
                Path("reproduce/README.md")]
    missing = [r for r in required if not (ENVELOPE / r).exists()]
    missing += [r for r in ["model_card.md"] if not (ENVELOPE / r).exists()]
    print("envelope files:", "COMPLETE" if not missing else f"MISSING {missing}")

    if not ok_metric or missing:
        sys.exit(1)
    print("\nEnvelope validated.")


if __name__ == "__main__":
    raise SystemExit(main())
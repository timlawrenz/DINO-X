#!/usr/bin/env python3
"""Generate a per-model data provenance catalog into an evidence envelope.

Emits `releases/{slug}/data_provenance.json` (machine-readable) and
`releases/{slug}/data_catalog.md` (human-readable) computed from the ACTUAL
on-disk index/label/split files — never hand-typed counts.

What it captures per model:
  - every dataset used (training index, fine-tune labels, external eval)
  - depth: slice counts, patient/series counts, spacing distribution
  - the exact index CSV path + a deterministic data-content hash
  - split manifests (train/val)
  - git commits + dataset-catalog hash (reusing zoo/lineage)
  - licensing/source per dataset

Usage:
    python scripts/release/generate_data_provenance.py \
      --model-slug dinox-ct-msd-hepatic-vessel-vit-base-v1 \
      --adapter-path adapters/msd-hepatic-vessel-vit-base-50k \
      [--train-index data/mvp/msd_hepatic_vessel_only_t2.csv] \
      [--val-labels data/msd-hepatic-vessel/labels/msd_hepatic_vessel_val.csv] \
      [--external-labels data/crlm/vessel_labels.csv] \
      [--catalog-dir zoo/datasets/ct]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def file_hash(path: Path, nbytes: int | None = None) -> str:
    """SHA-256 of a file's content (optionally first nbytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if nbytes:
            h.update(f.read(nbytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def catalog_hash(catalog_dir: Path) -> str:
    h = hashlib.sha256()
    if catalog_dir.is_dir():
        for y in sorted(catalog_dir.glob("*.yaml")):
            h.update(y.read_bytes())
    return h.hexdigest()[:16]


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def spacing_stats(rows: list[dict], x="spacing_x", y="spacing_y", z="spacing_z"):
    def num(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return None
    xs = [num(r, x) for r in rows if num(r, x) is not None]
    ys = [num(r, y) for r in rows if num(r, y) is not None]
    zs = [num(r, z) for r in rows if num(r, z) is not None]
    def stat(v):
        return {"min": round(min(v), 4), "max": round(max(v), 4),
                "mean": round(sum(v) / len(v), 4)} if v else {}
    return {"spacing_x": stat(xs), "spacing_y": stat(ys), "slice_thickness": stat(zs)}


def describe_index(index_csv: Path) -> dict:
    rows = csv_rows(index_csv)
    total = len(rows)
    by_dataset = Counter(r.get("dataset", "?") for r in rows)
    return {
        "path": str(index_csv),
        "content_hash_sha256_16": file_hash(index_csv),
        "total_slices": total,
        "by_dataset": dict(by_dataset),
        "spacing": spacing_stats(rows),
        "row_preview": rows[0] if rows else None,
        "row_keys": list(rows[0].keys()) if rows else [],
    }


def describe_labels(label_csv: Path, label_key="label", patient_key="patient_id") -> dict:
    rows = csv_rows(label_csv)
    pos = sum(1 for r in rows if r.get(label_key) in ("1", 1, "True", "true"))
    neg = len(rows) - pos
    n_patients = len({r.get(patient_key) for r in rows if r.get(patient_key)})
    return {
        "path": str(label_csv),
        "content_hash_sha256_16": file_hash(label_csv),
        "total_rows": len(rows),
        "positive": pos,
        "negative": neg,
        "positive_frac": round(pos / max(len(rows), 1), 4),
        "n_patients": n_patients,
        "row_keys": list(rows[0].keys()) if rows else [],
    }


def read_manifest(manifest_path: Path) -> dict:
    try:
        d = json.load(open(manifest_path))
    except Exception:
        return {}
    out = {}
    for k in ("train", "val"):
        v = d.get(k)
        if isinstance(v, list):
            out[k] = {"count": len(v), "sample": v[:3]}
        elif isinstance(v, dict) and "series_dir" in v:
            out[k] = {"count": len(v["series_dir"]), "sample": v["series_dir"][:3]}
    out["counts"] = d.get("counts", {})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", required=True)
    ap.add_argument("--adapter-path", required=True, type=Path)
    ap.add_argument("--train-index", type=Path, help="backbone pretraining index CSV")
    ap.add_argument("--finetune-train", type=Path, help="LoRA fine-tune train labels CSV")
    ap.add_argument("--val-labels", type=Path, help="LoRA fine-tune val labels CSV")
    ap.add_argument("--test-labels", type=Path, help="hold-out/adjacent test labels CSV (optional)")
    ap.add_argument("--external-labels", type=Path, help="external eval labels CSV (e.g. CRLM)")
    ap.add_argument("--manifest", type=Path, help="split_manifest json (optional)")
    ap.add_argument("--external-manifest", type=Path, help="external split/source manifest (optional)")
    ap.add_argument("--catalog-dir", type=Path, default=REPO_ROOT / "zoo" / "datasets" / "ct")
    args = ap.parse_args(argv)

    env = REPO_ROOT / "releases" / args.model_slug
    if not env.is_dir():
        print(f"ERROR: envelope {env} not found. Run build_envelope.py build first.")
        return 1

    provenance = {
        "model": args.model_slug,
        "generated": __file__.split("/")[-1],
        "git_commit": git_commit(REPO_ROOT),
        "data_catalog_hash_sha256_16": catalog_hash(args.catalog_dir),
        "datasets": [],
        "pretraining_training_index": None,
        "fine_tuning_labels": {},
        "external_validation": {},
    }

    # Dataset licensing map from the YAML catalog (best-effort)
    dataset_meta = {}
    if args.catalog_dir.is_dir():
        import yaml
        for y in sorted(args.catalog_dir.glob("*.yaml")):
            try:
                d = yaml.safe_load(open(y)) or {}
                name = d.get("name") or d.get("id") or y.stem
                dataset_meta[name.lower()] = {
                    "source": d.get("source") or d.get("source_url"),
                    "license": d.get("license"),
                    "redistribution": d.get("redistribution"),
                    "source_url": d.get("source_url"),
                    "citation": d.get("citation"),
                    "attribution": d.get("attribution"),
                }
            except Exception:
                pass

    # 1. pretraining index
    if args.train_index and args.train_index.exists():
        idx = describe_index(args.train_index)
        idx["manifest"] = read_manifest(args.manifest) if args.manifest and args.manifest.exists() else None
        idx["datasets"] = [
            {"name": k,
             **dataset_meta.get(k.lower(), {"source": "?", "license": "?"}),
             "slices_in_index": v}
            for k, v in idx["by_dataset"].items()
        ]
        provenance["pretraining_training_index"] = idx
        provenance["datasets"] = idx["datasets"]

    # 2. fine-tuning labels
    for key, path in [("finetune_train", args.finetune_train),
                      ("finetune_val", args.val_labels),
                      ("test", args.test_labels)]:
        if path and path.exists():
            provenance["fine_tuning_labels"][key] = describe_labels(path)

    # 3. external validation labels
    if args.external_labels and args.external_labels.exists():
        ext = describe_labels(args.external_labels)
        ext["manifest"] = read_manifest(args.external_manifest) if args.external_manifest and args.external_manifest.exists() else None
        ext["source_note"] = "External held-out set, disjoint from training (see the_science/03_validation.md)"
        provenance["external_validation"] = ext

    # write
    (env / "data_provenance.json").write_text(json.dumps(provenance, indent=2))
    (env / "data_catalog.md").write_text(render_md(provenance, dataset_meta))
    print(f"Wrote {env / 'data_provenance.json'}")
    print(f"Wrote {env / 'data_catalog.md'}")
    return 0


def render_md(p: dict, dataset_meta: dict) -> str:
    L = ["# Data Catalog & Provenance\n",
         f"**Model:** `{p['model']}`  \n",
         f"**Git commit:** `{p.get('git_commit')}`  \n",
         f"**Dataset catalog SHA-256 (16):** `{p.get('data_catalog_hash_sha256_16')}`\n",
         "\n> Generated from the actual on-disk index/label files, not hand-typed counts. "
         "Machine-readable twin: `data_provenance.json`.\n"]

    L.append("\n## 1. Datasets Involved\n")
    L.append("| Dataset | Source | License | Redistribution | Slices-in-index |")
    L.append("|---|---|---|---|---|")
    for d in p.get("datasets", []):
        L.append(f"| {d.get('name','?')} | {d.get('source','?')} | {d.get('license','?')} | "
                 f"{d.get('redistribution','?')} | {d.get('slices_in_index','?')} |")

    if p.get("pretraining_training_index"):
        idx = p["pretraining_training_index"]
        L.append("\n## 2. Pretraining Training Index\n")
        L.append(f"- **Path:** `{idx['path']}`")
        L.append(f"- **Content SHA-256 (16):** `{idx['content_hash_sha256_16']}`")
        L.append(f"- **Total slices:** {idx['total_slices']}")
        for ds, n in idx["by_dataset"].items():
            L.append(f"  - `{ds}`: {n} slices")
        L.append("\n**Spacing distribution:**")
        L.append("```json")
        L.append(json.dumps(idx.get("spacing", {}), indent=2))
        L.append("```")
        if idx.get("manifest"):
            L.append("\n**Split manifest:**")
            L.append("```json")
            L.append(json.dumps(idx["manifest"], indent=2))
            L.append("```")

    if p.get("fine_tuning_labels"):
        L.append("\n## 3. Fine-Tuning Labels\n")
        for k, v in p["fine_tuning_labels"].items():
            L.append(f"\n**{k}** (`{v['path']}` hash `{v['content_hash_sha256_16']}`):")
            L.append(f"- rows: {v['total_rows']}, positive: {v['positive']} ({v['positive_frac']}), "
                     f"negative: {v['negative']}, patients: {v['n_patients']}")

    if p.get("external_validation"):
        v = p["external_validation"]
        L.append("\n## 4. External Validation\n")
        L.append(f"- **Path:** `{v['path']}` hash `{v['content_hash_sha256_16']}`")
        L.append(f"- **Slices:** {v['total_rows']}, patients: {v['n_patients']}, "
                 f"positive: {v['positive']} ({v['positive_frac']})")
        L.append(f"- **Note:** {v.get('source_note', '')}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
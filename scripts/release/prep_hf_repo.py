#!/usr/bin/env python3
"""Prep a model + its full evidence envelope for HuggingFace Hub upload (private by default).

Stages everything a release needs into a local directory so it can be reviewed
before (or without) uploading:
  - frozen backbone (safetensors + .pth + config.json)   via zoo/hub export
  - LoRA adapter (adapter_model.safetensors + adapter_config.json)
  - task head (head.pth) + finetune_config.json
  - the complete evidence envelope from releases/{slug}/ (the_science/, data_catalog,
    data_provenance, evaluation, reproduce, model_card)
  - a README.md (HF model card) assembled from the envelope

Usage (staging only, no token):
    python scripts/release/prep_hf_repo.py \
      --model-slug dinox-ct-msd-hepatic-vessel-vit-base-v1 \
      --adapter-path adapters/msd-hepatic-vessel-vit-base-50k \
      --out-dir /tmp/hf-preview/dinox-...

To also create a PRIVATE repo on HF (needs HF_TOKEN):
  --upload --repo-id timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1 [--private]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-slug", required=True)
    ap.add_argument("--adapter-path", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="local staging dir for the release contents")
    ap.add_argument("--upload", action="store_true", help="create+upload private HF repo")
    ap.add_argument("--repo-id", default=None, help="e.g. timlawrenz/dinox-...")
    ap.add_argument("--private", action="store_true", default=True, help="private repo")
    ap.add_argument("--token", default=None, help="HF token (or HF_TOKEN env)")
    args = ap.parse_args(argv)

    env = REPO_ROOT / "releases" / args.model_slug
    adapter = args.adapter_path
    if not env.is_dir():
        print(f"ERROR: envelope {env} not found — run build_envelope.py + generate_data_provenance.py first")
        return 1
    if not (adapter / "finetune_config.json").exists():
        print(f"ERROR: adapter {adapter} missing finetune_config.json")
        return 1

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Frozen backbone (from the finetune_config backbone checkpoint) ----
    ft = json.load(open(adapter / "finetune_config.json"))
    from zoo.hub import load_from_training_checkpoint, export_hub_checkpoint
    backbone_ckpt = Path(ft["backbone"])
    if not backbone_ckpt.exists():
        print(f"WARN: backbone checkpoint {backbone_ckpt} not found; skipping backbone export")
    else:
        backbone = load_from_training_checkpoint(backbone_ckpt, device="cpu")
        export_hub_checkpoint(backbone, output_dir=str(out), use_safetensors=True)
        export_hub_checkpoint(backbone, output_dir=str(out), use_safetensors=False)  # .pth compat
        print(f"Exported backbone -> {out}/backbone.safetensors")

    # ---- 2. LoRA adapter + head + configs ----
    for fname in ["adapter_model.safetensors", "adapter_config.json", "head.pth",
                  "finetune_config.json"]:
        f = adapter / fname
        if f.exists():
            shutil.copy(f, out / fname)
            print(f"Copied {fname}")
        else:
            print(f"WARN: adapter missing {fname}")

    # ---- 3. Evidence envelope ----
    # copy the_science/, data files, evaluation, provenance, reproduce, model_card
    for sub in ["the_science", "reproduce", "experiments"]:
        src = env / sub
        if src.is_dir():
            shutil.copytree(src, out / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ["data_catalog.md", "data_provenance.json", "evaluation.json",
              "provenance.yaml", "model_card.md"]:
        src = env / f
        if src.exists():
            shutil.copy(src, out / f)

    # ---- 4. README.md: model card (+ envelope pointers) ----
    card = (env / "model_card.md").read_text() if (env / "model_card.md").exists() else ""
    readme = card + "\n\n---\n\n# Evidence Envelope\nThis repository ships the full science trail, not just weights. See:\n\n"
    for name in ["the_science/01_decision_trail.md",
                 "the_science/02_training_recipe.md",
                 "the_science/03_validation.md",
                 "the_science/04_negative_results.md",
                 "the_science/05_known_limits.md",
                 "data_catalog.md", "reproduce/README.md"]:
        if (out / name).exists():
            readme += f"- [{name}]({'/' + name if False else name})\n"
    (out / "README.md").write_text(readme)

    # ---- 5. upload (optional) ----
    if args.upload:
        import os
        from huggingface_hub import HfApi
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print("ERROR: --upload requires a token (--token or HF_TOKEN)")
            return 1
        repo_id = args.repo_id or f"timlawrenz/{args.model_slug}"
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(out), repo_id=repo_id,
                          commit_message=f"Release {args.model_slug} + evidence envelope (private preview)")
        url = f"https://huggingface.co/{repo_id}"
        print(f"Uploaded to {url}")
    else:
        print("\nStaged (no upload). Contents:")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(out)}  ({p.stat().st_size // 1024} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
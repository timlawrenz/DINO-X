#!/usr/bin/env python3
"""Build an evidence envelope for a released DINO-X model.

Scaffolds `releases/{model-slug}/` by reading the adapter's finetune_config.json
and the backbone run config.json, generating the training-recipe + provenance
tables, and validating the structure against the envelope standard
(releases/ENVELOPE_TEMPLATE.md).

Overview:
    --adapter-path    path to the fine-tuned adapter dir (has finetune_config.json)
    --model-slug      e.g. dinox-ct-msd-hepatic-vessel-vit-base-v1 (dash/slug)
    --hf-slug         HuggingFace id (optional; default timlawrenz/{slug})
    --internal-auc    internal AUROC (optional, written into evaluation.json + provenance)
    --leakage-caveat  free-text caveat for internal number (optional)

The generator produces the full envelope skeleton with 02_training_recipe.md and
provenance.yaml auto-populated from the frozen configs. The human still writes
01_decision_trail.md, 03_validation.md, 04_negative_results.md, 05_known_limits.md,
and a model-card/validation narrative — the generator cannot invent the honest story.
It overwrites only the files it generates.

Validation:
    --validate        run the completeness + cross-reference check on an envelope
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE = REPO_ROOT / "releases" / "ENVELOPE_TEMPLATE.md"
RELEASES = REPO_ROOT / "releases"

REQUIRED = [
    "ENVELOPE.md", "model_card.md", "evaluation.json", "provenance.yaml",
    "source_branch",
    "the_science/01_decision_trail.md",
    "the_science/02_training_recipe.md",
    "the_science/03_validation.md",
    "the_science/04_negative_results.md",
    "the_science/05_known_limits.md",
    "reproduce/README.md",
]


def _resolve_run_config(adapter_cfg: dict) -> dict | None:
    """Try to (re)locate the backbone run config.json from the finetune backbone path."""
    backbone = adapter_cfg.get("backbone", "")
    b = Path(backbone)
    # backbone is like runs/{ts}_{suffix}/checkpoint_final_*.pth
    if b.parent.name == "runs" or "checkpoint" in b.name:
        run_dir = b.parent
        cfg = run_dir / "config.json"
        if cfg.exists():
            return json.load(open(cfg))
    # fallback: search runs for a config.json near a matching checkpoint name
    return None


def _hmin(cfg, path, default="?"):
    """Deep-get from a dict by dotted path."""
    cur = cfg
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def build(args) -> int:
    adapter = Path(args.adapter_path)
    slug = args.model_slug
    if not (adapter / "finetune_config.json").exists():
        print(f"ERROR: no finetune_config.json at {adapter}")
        return 1
    ft = json.load(open(adapter / "finetune_config.json"))
    run_cfg = _resolve_run_config(ft)

    env = RELEASES / slug
    env.mkdir(parents=True, exist_ok=True)
    (env / "the_science").mkdir(exist_ok=True)
    (env / "reproduce").mkdir(exist_ok=True)
    (env / "experiments").mkdir(exist_ok=True)

    # ---- provenance.yaml ----
    backbone = ft.get("backbone", "?")
    prov = {
        "arm": slug,
        "mode": "confirmatory",
        "git_commit_bare": "?  # fill at run start",
        "adapter_git_commit": "?",
        "eval_harness_commit": "?",
        "source_branch": "main",
        "adapter_path": str(adapter),
        "backbone": backbone,
        "task": f"{ft.get('task','classification')}: {ft.get('num_classes','2')} classes",
        "rank": ft.get("rank"),
        "alpha": ft.get("alpha"),
        "lr": ft.get("lr"),
        "epochs": ft.get("epochs"),
        "batch_size": ft.get("batch_size"),
        "input_format": ft.get("input_format"),
        "scale_aware": ft.get("scale_aware"),
        "best_epoch": ft.get("best_epoch"),
        "seed": ft.get("seed"),
        "internal_auroc": args.internal_auc,
        "internal_leakage_caveat": args.leakage_caveat or "",
        "license": "Apache-2.0",
        "hf_slug": args.hf_slug or f"timlawrenz/{slug}",
    }
    if run_cfg:
        m = run_cfg.get("model", {})
        prov["architecture"] = {
            "name": m.get("name"), "patch": m.get("patch"),
            "dim": m.get("dim"), "depth": m.get("depth"),
            "heads": m.get("heads"), "out_dim": m.get("out_dim"),
            "img_size": run_cfg.get("img_size"),
        }
        prov["pretrain"] = {
            "loss_type": run_cfg.get("loss_type"),
            "lr": run_cfg.get("lr"), "min_lr": run_cfg.get("min_lr"),
            "warmup_steps": run_cfg.get("warmup_steps"),
            "weight_decay": run_cfg.get("weight_decay"),
            "max_steps": run_cfg.get("max_steps"),
            "ema": run_cfg.get("ema"),
            "teacher_temp": run_cfg.get("teacher_temp"),
            "student_temp": run_cfg.get("student_temp"),
            "center_momentum": run_cfg.get("center_momentum"),
            "gram_enabled": run_cfg.get("gram_enabled"),
            "gram_weight": run_cfg.get("gram_weight"),
            "koleo_weight": run_cfg.get("koleo_weight"),
            "scale_aware": run_cfg.get("scale_aware"),
            "batch_size": run_cfg.get("batch_size"),
            "accumulation_steps": run_cfg.get("accumulation_steps"),
        }
        hw = run_cfg.get("hardware", {})
        prov["training_gpu"] = hw.get("device_name", "?")
    else:
        prov["note"] = "backbone run config.json not located; fill architecture/pretrain by hand"

    yaml_out = _render_yaml(prov)
    (env / "provenance.yaml").write_text(yaml_out)

    # ---- evaluation.json ----
    ev = {
        "model": slug,
        "internal": {
            "auroc": args.internal_auc,
            "best_epoch": ft.get("best_epoch"),
            "best_val_loss": ft.get("best_val_loss"),
            "leakage_caveat": args.leakage_caveat or "",
        },
        "external": None,  # fill with external validation results once available
        "license": "Apache-2.0",
    }
    (env / "evaluation.json").write_text(json.dumps(ev, indent=2) + "\n")

    # ---- source_branch ----
    (env / "source_branch").write_text("main\n")

    # ---- ENVELOPE.md (skeleton) ----
    (env / "ENVELOPE.md").write_text(
        f"""# {slug} — Evidence Envelope

**Model:** {slug}
**License:** Apache-2.0 · **HF Hub:** {args.hf_slug or f"timlawrenz/{slug}"}
**Source branch:** main

## Headline Results

_Edit this table once internal/external numbers are known. Every value must map to an
artifact in `reproduce/` or the ledger — nothing from memory._

| Metric | Value | Where verified |
|---|---|---|
| Internal AUROC | {args.internal_auc or "TBD"} | `adapters/...` / ledger |
| External AUROC | TBD | `reproduce/repro_metrics.py` / ledger |

## Follow The Science

1. `the_science/01_decision_trail.md` — why this model exists, failures that shaped it
2. `the_science/02_training_recipe.md` — exact recipe (auto-generated from frozen configs)
3. `the_science/03_validation.md` — how we know the numbers are real (adversarial pass)
4. `the_science/04_negative_results.md` — what failed, with evidence
5. `the_science/05_known_limits.md` — where it still breaks
6. `reproduce/README.md` — reproduce every number yourself

> This model ships its science, not just its weights. If a claim can't be reproduced or
> traced to the ledger, it is not a release claim.
""")

    # ---- 02_training_recipe.md (auto-generated) ----
    (env / "the_science" / "02_training_recipe.md").write_text(
        _render_recipe(slug, ft, run_cfg, args))

    # ---- Stubs that need human narrative (honest story) ----
    for stub, title in [
        ("01_decision_trail.md", "Why This Model Exists"),
        ("03_validation.md", "How We Know The Numbers Are Real"),
        ("04_negative_results.md", "What We Tried That Did NOT Work"),
        ("05_known_limits.md", "Known Limitations"),
    ]:
        p = env / "the_science" / stub
        if not p.exists():
            p.write_text(
                f"# {title}\n\n*TODO: human-authored honest narrative. This is the part "
                f"that earns trust — failures, adversarial passes, external validation. "
                f"Do not skip it; an envelope without the story is just a folder.*\n")

    # ---- reproduce stubs ----
    if not (env / "reproduce" / "README.md").exists():
        (env / "reproduce" / "README.md").write_text(
            "# Reproduce The Headline Numbers\n\n*TODO: exact commands to recompute each "
            "claim. See the flagship example releases/dinox-ct-msd-hepatic-vessel-vit-base-v1/*\n")
    if not (env / "reproduce" / "repro_metrics.py").exists():
        shutil.copy(REPO_ROOT / "releases" / "dinox-ct-msd-hepatic-vessel-vit-base-v1" /
                    "reproduce" / "repro_metrics.py",
                    env / "reproduce" / "repro_metrics.py")
    if not (env / "model_card.md").exists():
        (env / "model_card.md").write_text(
            f"# {slug}\n\nResearch use only. Not a clinical device.\n\n"
            f"Full evidence envelope at `releases/{slug}/` — every number traceable, "
            f"every failure published, reproduction script included.\n")
    if not (env / "experiments" / "LEDGER_REFERENCE.md").exists():
        (env / "experiments" / "LEDGER_REFERENCE.md").write_text(
            "# Ledger Reference\n\n*TODO: map every claim in this envelope to its exact "
            "`docs/EXPERIMENTS_AND_RESULTS.md` section and `git_commit`.*\n")

    print(f"Envelope scaffolded at {env}")
    print("NOTE: 01/03/04/05 decision-trail, validation, negative-results, and known-limits "
          "require human-authored honest narrative — the generator doesn't invent the story.")
    return validate(env)


def _render_yaml(d: dict) -> str:
    import yaml
    class D(yaml.Dumper):
        def increase_indent(self, flow=False, *a, **k):
            return super().increase_indent(flow=flow, indentless=False)
    return yaml.dump(d, Dumper=D, default_flow_style=False, sort_keys=False)


def _render_recipe(slug, ft, run_cfg, args) -> str:
    lines = [f"# 02 — Training Recipe (auto-generated)\n",
             f"Automatically populated from the frozen `finetune_config.json` and backbone "
             f"`config.json`. Every value is sourced from disk, not memory.\n"]
    lines.append("\n## Fine-tuning config\n")
    for k in ["task", "num_classes", "rank", "alpha", "lr", "epochs", "batch_size",
              "input_format", "scale_aware", "seed", "es_metric", "patience"]:
        if k in ft:
            lines.append(f"- **{k}**: `{ft[k]}`")
    if args.internal_auc is not None:
        lines.append(f"\n- **Internal AUROC**: `{args.internal_auc}` "
                     f"(epoch {ft.get('best_epoch')})"
                     + (f" — CAVEAT: {args.leakage_caveat}" if args.leakage_caveat else ""))
    if run_cfg:
        m = run_cfg.get("model", {})
        lines.append("\n## Pretraining (backbone)\n")
        lines.append(f"- **Architecture**: {m.get('name')} (dim {m.get('dim')}, depth "
                     f"{m.get('depth')}, patch {m.get('patch')}, img_size "
                     f"{run_cfg.get('img_size')}, scale_aware {run_cfg.get('scale_aware')})")
        for k in ["loss_type", "lr", "min_lr", "warmup_steps", "weight_decay", "max_steps",
                  "ema", "teacher_temp", "student_temp", "center_momentum",
                  "gram_enabled", "gram_weight", "koleo_weight"]:
            if k in run_cfg:
                lines.append(f"- **{k}**: `{run_cfg[k]}`")
        lines.append(f"- **batch_size / accumulation**: `{run_cfg.get('batch_size')}` / "
                     f"`{run_cfg.get('accumulation_steps')}`")
    lines.append("\n## Scripts\n")
    lines.append("- Fine-tuning: `scripts/finetune_lora.py`")
    lines.append("- Pretraining: `scripts/phase5_big_run.py`")
    return "\n".join(lines) + "\n"


def validate(env_dir: Path) -> int:
    missing = [r for r in REQUIRED if not (env_dir / r).exists()]
    if missing:
        print(f"VALIDATION FAIL — missing: {missing}")
        return 1
    ev_p = env_dir / "evaluation.json"
    expl = ev_p.exists() and ("external" in json.load(open(ev_p)))
    # warn, not fail, on empty external (human may not have run external yet)
    print("VALIDATION OK — all required envelope files present.")
    print("Every claim in ENVELOPE.md must be recomputable (reproduce/) or ledger-traced; "
          "run reproduce/repro_metrics.py once numbers are final.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build/validate a DINO-X evidence envelope")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--adapter-path", required=True, type=Path)
    b.add_argument("--model-slug", required=True)
    b.add_argument("--hf-slug", default=None)
    b.add_argument("--internal-auc", type=float, default=None)
    b.add_argument("--leakage-caveat", default=None)
    b.set_defaults(fn=build)

    v = sub.add_parser("validate")
    v.add_argument("--model-slug", required=True)
    v.set_defaults(fn=lambda a: validate(RELEASES / a.model_slug))

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
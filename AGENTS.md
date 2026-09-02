# AGENTS.md — DINO-X

AI agent entry point. Read these docs in order before touching anything. The governance docs are the source of truth; this file is the index.

---

## Mandatory Reading Order

1. **`PROJECT_STATUS.md`** — Living orientation: current phase, blockers, next action, headline result.
2. **`docs/EXPERIMENT_TREE.md`** — Shallow workstream map with status tags ([ACTIVE]/[CONCLUDED]/[TBD]).
3. **`docs/EXPERIMENTS_AND_RESULTS.md`** — Permanent ledger: all empirical findings with pre-registered gates, adversarial pass checklists, and GO/PIVOT/PARK/KILL verdicts.
4. **This file (`AGENTS.md`)** — Critical rules and project-specific conventions.
5. **Load the skills** — The source of truth for *how* experiments run here lives in the skills, not in repo governance docs. Load `scientific-experiment-structure` (data/measurement design gates, adversarial pass, leakage rules) and `autonomous-research-execution` (orchestration) before planning or running any experiment. The skills evolve faster than any per-repo doc; follow them over `docs/experiment-structure.md`, which may be stale.

Documentation lives in `docs/`. Repo governance: `docs/experiment-structure.md` (may lag the skills — use as reference, not authority).

---

## Critical Rules — Never Violate These

1. **Never re-run a `[CONCLUDED — FAIL]` or `[CONCLUDED — KILL]` experiment** without explicit user direction and a new hypothesis. Negative results are documented to prevent re-attempt.
2. **Never skip the adversarial pass.** Before writing any PASS verdict in the ledger, complete the 4-question checklist (see `docs/experiment-structure.md`). A trusted PASS that was actually a measurement bug is the most expensive failure mode.
3. **Always read `PROJECT_STATUS.md` first.** The project may be PARKed or between phases.
4. **Always verify identity/label test sets visually.** Contamination can silently produce numbers that look right and are wrong everywhere that matters.
5. **Config keys not consumed by code produce silently invalid experiments.** grep-trace every key through the codebase before launching.
6. **`git_commit` must be recorded at run start.** Without a commit SHA, the experiment is not reproducible.
7. **Reproducibility and clear documentation are as important as successful training.** Where possible, infrastructure as code, automated tests, and clear documentation should be provided.

---

## Project-Specific Conventions

### Hardware

| Host | GPU | Memory | Use |
|---|---|---|---|
| `game` | RTX 4090 / 3090 Ti | 24GB | Cloud / CUDA experiments |
| `max395` | AMD Radeon 8060S (Strix Halo) | 128GB unified (~96GB VRAM) | Large model training |
| Various | RTX 2070 SUPER | 8GB | Local validation |

### Storage

- **NAS:** `/mnt/nas-ai-models/training-data/dino-x/` — datasets, labels, adapter checkpoints
- **HF Hub:** `timlawrenz/dinox-mvp-data` (processed data), planned: `timlawrenz/dinox-ct-vit-small-v1`
- **runs/:** Training artifacts (checkpoints, TensorBoard, configs) — in git but large files via LFS

### Strix Halo (ROCm) Environment

Always source before training:
```bash
source scripts/rocm_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Key ROCm env vars from the env script:
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- `PYTORCH_TUNABLEOP_ENABLED=1` + `PYTORCH_TUNABLEOP_TUNING=1`
- `TORCH_BLAS_PREFER_HIPBLASLT=1`
- `PYTHONUNBUFFERED=1`

**Known issue:** ViT-Large (923M params) saturates LPDDR5 bandwidth. Mitigations:
- `--grad-checkpoint` — trade math for memory footprint
- `--batch-size 4 --accumulation-steps 64` — smaller physical batch fits Infinity Cache/L2
- Without these, throughput drops to ~1 img/s (126-day ETA). With them: ~26 img/s (5.6-day ETA).

### Script-Driven Experiments

DINO-X uses CLI-driven scripts, not config-file-driven training loops:

- **Pretraining:** `scripts/phase5_big_run.py` (DINO + Gram + ScaleEmbedding)
- **Fine-tuning:** `scripts/finetune_lora.py` (LoRA adapter training)
- **Evaluation:** `scripts/evaluate_panorgan.py` (6-metric suite), `scripts/phase5_view_retrieval_eval.py`
- **Preprocessing:** `scripts/preprocessing/phase2_preprocess_lidc_idri.py`, `scripts/preprocessing/mvp_combine_indices.py`

Configs in `configs/` (e.g., `panorgan_ct_vits.yaml`) are staging specs, not runtime source of truth. Runtime parameters are CLI args + frozen `config.json` per run.

### LoRA Fine-Tuning Standard Protocol

- **Backbone:** Frozen ViT-Small/Large with ScaleEmbedding (always frozen)
- **LoRA:** rank=8, alpha=16, targets=qkv+proj+fc1+fc2
- **Task head:** Outside PEFT — saved separately from adapter weights
- **Data:** 64px nodule crops, lung HU window (level=-30, width=120 scaled; real: level=-300, width=1200)
- **Training:** 50 epochs, batch=32, lr=5e-4, early stopping on val AUROC (patience=10), `--seed 42`
- **Script:** `scripts/finetune_lora.py`
- **Output:** `adapters/{slug}/` — adapter_model.safetensors, head.pth, adapter_config.json, finetune_config.json

### Experiment Naming

- **Runs:** Timestamp-based: `runs/{YYYYMMDD_HHMMSS}_{run-suffix}/`
- **Adapters:** Descriptive: `adapters/{task}-{backbone}-{variant}/`
- **Arm slugs (governance):** Descriptive lowercase-hyphen: `5dataset-phase3-small-bs256`, `lidc-malignancy-lora-r8-64px-lung-window`

### Testing

186 tests in `tests/`. Run with pytest. Key validation script: `scripts/integration_canary.py` (infrastructure verification).

### Codebase (zoo/ package)

- `zoo/arch.py` — PatchViT + ScaleEmbedding architecture
- `zoo/data.py` — Unified CT data loader (ManifestDataset)
- `zoo/peft.py` — LoRA adapter inject/save/load
- `zoo/hub.py` — Model loading (local / HuggingFace Hub)
- `zoo/lineage.py` — Training provenance tracking
- `zoo/registry.py` — YAML dataset catalog
- `zoo/manifest.py` — Parquet per-slice metadata
- `zoo/encode.py` — Zero-preprocessing inference API
- `zoo/card.py` / `zoo/publish.py` — HF model card generation + publishing
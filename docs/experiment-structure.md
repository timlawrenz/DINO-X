# Experiment Structure — DINO-X

Governance document describing how this project implements the Scientific Experiment Structure. Tailored to DINO-X's conventions. The canonical skill is at `~/.hermes/profiles/dino-x/skills/scientific-experiment-structure/SKILL.md`.

---

## Directory Structure

```
DINO-X/
├── PROJECT_STATUS.md               # Living orientation pointer (always read first)
├── AGENTS.md                       # AI agent entry point: mandatory reading order + critical rules
├── README.md                       # Human-facing overview
├── docs/
│   ├── experiment-structure.md     # This governance document (in git)
│   ├── EXPERIMENT_TREE.md          # Living workstream map (active/concluded/TBD with verdicts)
│   ├── EXPERIMENTS_AND_RESULTS.md  # Permanent ledger: dated findings with pre-registered gates + adversarial pass
│   ├── DISCONTINUATION_NOTICE_*.md # Tombstones for KILLed approaches
│   └── ...                         # Other project docs (roadmap, hardware, data)
├── configs/                        # Staging specs (not runtime source of truth)
│   └── panorgan_ct_vits.yaml       # Staged pan-organ CT training spec (stages A/B/C)
├── runs/                           # Training artifacts (checkpoints, TensorBoard, configs)
│   └── {YYYYMMDD_HHMMSS}_{suffix}/ # Timestamped run directory
│       ├── config.json             # Frozen runtime config
│       ├── checkpoint_*.pth        # Model checkpoints
│       ├── events.out.tfevents.*   # TensorBoard event files
│       └── view_retrieval_*.json   # Evaluation results
├── adapters/                       # LoRA fine-tuned adapters
│   └── {task}-{backbone}-{variant}/ # Descriptive adapter directory
│       ├── adapter_model.safetensors
│       ├── head.pth
│       ├── adapter_config.json
│       ├── finetune_config.json
│       └── README.md               # Placeholder (model card template)
├── zoo/                            # Model zoo package (source code)
├── scripts/                        # Training, evaluation, and preprocessing scripts
├── tests/                          # 186 tests (pytest)
├── data/                           # Preprocessed datasets (symlinked to NAS)
└── requirements.txt
```

## Project-Specific Divergences from Canonical Skill

### Script-Driven Experiments

DINO-X is script-driven, not config-file-driven. The canonical skill prescribes `experiments/{arm-slug}/config.yaml` + `train.py --config`. DINO-X uses CLI-driven scripts:

- **Pretraining:** `scripts/phase5_big_run.py` with extensive CLI arguments
- **Fine-tuning:** `scripts/finetune_lora.py`
- **Evaluation:** `scripts/evaluate_panorgan.py`, `scripts/phase5_view_retrieval_eval.py`

Config files (`configs/panorgan_ct_vits.yaml`) are staging specs — they describe intended runs but are not the runtime source of truth. The frozen `config.json` in each `runs/{timestamp}/` directory is the immutable record.

### Timestamp-Based Run Naming

Instead of `experiments/{arm-slug}/`, DINO-X uses `runs/{YYYYMMDD_HHMMSS}_{run-suffix}/`. This is a valid divergence — the timestamp is an immutable identifier, and `--run-suffix` provides semantic naming. Arm slugs for governance documents are abbreviated descriptive names: `5dataset-phase3-small-bs256`, `two-organ-scale-aware`.

### Per-Arm Retrofits for Concluded Work

For concluded script-driven work that pre-dates this governance structure, use **suffixed filenames** in a shared directory rather than creating subdirectories. Example:

```
experiments/dino-x-v1/
├── provenance_two-organ-scale-aware.yaml
├── config_two-organ-scale-aware.yaml
├── provenance_5dataset-bs128.yaml
├── config_5dataset-bs128.yaml
├── provenance_lora-benchmark.yaml
├── config_lora-benchmark.yaml
└── ...
```

The arm-slug embedded in the filename is the canonical identifier in the tree and ledger. For **new** work starting after this governance structure, create a proper `experiments/{arm-slug}/` directory.

### Adapters vs Experiments

DINO-X has two kinds of experimental output:
- **`runs/`**: Pretraining artifacts (backbone checkpoints, TensorBoard, raw training configs)
- **`adapters/`**: LoRA fine-tuning artifacts (adapter weights, task heads, finetune configs)

A single backbone checkpoint can drive many adapters — the 5-dataset bs256 50K checkpoint has 5+ adapters at different LRs and checkpoints. The adapter naming convention embeds the backbone reference in the directory name.

## Storage

- **NAS:** `/mnt/nas-ai-models/training-data/dino-x/` — datasets, raw labels, adapter checkpoints
- **HF Hub:** `timlawrenz/dinox-mvp-data` (processed PNG slices + manifests)
- **Future HF Hub:** `timlawrenz/dinox-ct-vit-small-v1` (published backbone)
- **runs/ in git:** Checkpoint files handled via Git LFS if needed; config.json and TensorBoard events stored directly

## Three-Document System (DINO-X Implementation)

The three governance documents are:

| Document | Path | Role |
|---|---|---|
| Tree | `docs/EXPERIMENT_TREE.md` | Shallow map of all workstreams with status tags ([ACTIVE]/[CONCLUDED]/[TBD]) + verdicts (GO/PIVOT/PARK/KILL) |
| Ledger | `docs/EXPERIMENTS_AND_RESULTS.md` | Permanent record of empirical findings with pre-registered gates, adversarial pass checklists, and verdicts |
| Status | `PROJECT_STATUS.md` | Living orientation: current phase, blockers, next action, headline result |

The current `docs/EXPERIMENTS.md` functions as a combined ledger — it will be renamed/restructured into `EXPERIMENTS_AND_RESULTS.md` with the structured format (pre-registered gates, adversarial pass checklists, GO/PIVOT/PARK/KILL verdicts).

## Pre-Registered Gates

Before running any experiment, state the pass/fail criteria explicitly:

```markdown
**Pre-registered gate (stated BEFORE results):**
> PASS if val AUROC > 0.685 AND training converges without NaN.
> FAIL if val AUROC < 0.670 OR loss diverges after epoch 20.
```

## Adversarial Pass (4-Question Checklist)

**Rule: never write PASS in the ledger until this checklist is complete.** A trusted PASS that was actually a measurement bug is the most expensive failure mode.

1. Is the metric's own code tested? (validator, scorer, eval harness has unit tests)
2. Has the metric definition stayed stable across the runs being compared? (if not, re-run baseline)
3. Is the result reproducible? (re-run with different seed or fresh process)
4. Do the extremes and edge cases look right? (eyeball top/bottom/dead-center predictions)

```markdown
**Adversarial pass (fill BEFORE writing the verdict):**
- [ ] Metric code (validator/scorer/harness) has unit tests — commit: ______
- [ ] Metric definition unchanged vs compared arms (or baseline re-run) — version: ______
- [ ] Result reproduced (2nd seed / fresh process) — run: ______
- [ ] Extremes + edge cases inspected — artifact: ______
Verdict: PASS / FAIL / PENDING   (PENDING if any box is unchecked)
```

## Verdict Vocabulary

| Verdict | Meaning | Required Artifact |
|---|---|---|
| **GO** | Hypothesis held; continue / scale / productionize | Ledger entry with PASS + adversarial pass complete |
| **PIVOT** | Core idea partially works; redirect to the part that does | Ledger entry naming what worked vs what didn't + new direction |
| **PARK** | Inconclusive but not disproven; blocked on external input | PROJECT_STATUS.md naming exact unblock condition |
| **KILL** | Hypothesis disproven or approach fundamentally unsuitable | `docs/DISCONTINUATION_NOTICE_{slug}.md` (non-negotiable) |

## Process for Agents (AI and Human)

### Before running ANY experiment

1. Read `PROJECT_STATUS.md` — know the current phase, blockers, and the single next action.
2. Read `docs/EXPERIMENT_TREE.md` — check whether this experiment is already `[CONCLUDED]`. If it is, stop.
3. Check the ledger for prior related experiments.
4. Create the new run with `--run-suffix {arm-slug}`.
5. Record `git rev-parse HEAD` for provenance.
6. Register the arm in the tree as `[ACTIVE]`.

### When the experiment produces results

7. Run the adversarial pass on any candidate PASS.
8. Write the ledger entry with: Goal, Pre-registered gate, Empirical Evidence, Adversarial Pass checklist, Verdict.
9. Update the tree: move from `[ACTIVE]` to `[CONCLUDED — GO/PIVOT/PARK/KILL]`.
10. Update `PROJECT_STATUS.md` with the new headline result and next action.

### When an experiment is KILLed

11. Write `docs/DISCONTINUATION_NOTICE_{slug}.md` — the single most valuable artifact a dead experiment produces.
12. Record the KILL in the ledger and tree.
13. Never delete the code or artifacts.

## Hardware-Specific Rules

### Strix Halo (ROCm)

- Always source `scripts/rocm_env.sh` before training
- Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation
- ViT-Large (923M params) saturates LPDDR5 bandwidth — use `--grad-checkpoint` and small physical batch (4×64 or 8×32)
- Throughput sweet spot: ~26 img/s. Without mitigation: ~1 img/s (126-day ETA for 50K steps)
- bfloat16 AMP, no grad scaler needed
- ROCm JIT compilation on first step — expect slow initial throughput

### CUDA (RTX 4090 / 3090 Ti)

- fp16 AMP with grad scaler
- Effective batch 256 recommended for DINO stability
- 24GB VRAM typically fine for ViT-Small/Large with gradient checkpointing

### Testing & Validation

- `scripts/integration_canary.py` — automated infrastructure verification before long runs
- 186 tests in `tests/` — run with pytest
- Always run `scripts/integration_canary.py` after environment changes

## Common Pitfalls (DINO-X Specific)

- **Gradient accumulation step counting:** `phase5_big_run.py` counts micro-batches, not optimizer steps. With accum=32, warmup steps spread over 32× more micro-batches than expected.
- **ROCm allocator fragmentation:** Strix Halo crashes at ~5K effective steps for ViT-Large. `expandable_segments:True` is the primary mitigation.
- **Entropy wall:** DINO loss flatlining at ln(8192)=9.01 means the model isn't learning. Center momentum (0.999) + sharp teacher temp (0.02) breaks through it.
- **Feature collapse:** Teacher entropy approaching 0.0 means catastrophic collapse. Monitor entropy alongside loss — a model can have low loss and be collapsed.
- **Capacity dilution with ViT-Small:** 22M params cannot maintain organ-specific features across 5+ heterogeneous organ domains. The 4-dataset specialist (lung-focused) out-performs the 5-dataset generalist on lung-specific tasks.
- **Crop size dominates LoRA performance:** 64px >> 128px >> 224px for nodule classification. Task-specific HU windowing adds +3% AUROC.
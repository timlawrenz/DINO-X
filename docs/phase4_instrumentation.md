# Phase 4: Instrumentation (Navigation)

Phase 4 adds two safety rails before the Phase 5 multi-day run:

1) **A deterministic validation split** (10%) so we can monitor progress without leakage.
2) **A lightweight training monitor** that can be run periodically (e.g., daily) alongside training.

## 1) Create the validation split manifest

The split operates on the Phase 2 index (`data/processed/_index/index.csv`) and splits at `series_dir` granularity.

```bash
python3 scripts/phase4_make_split_manifest.py \
  --index-csv data/processed/_index/index.csv \
  --seed 0 \
  --val-frac 0.10
```

Default output:
- `data/processed/_splits/val10_seed0.json`

## 2) Use the split in training

Phase 3 micro-run supports excluding the validation set via `--split-manifest`:

```bash
python3 scripts/phase3_micro_run.py \
  --split-manifest data/processed/_splits/val10_seed0.json
```

## 3) Run the monitor

If you're using ROCm PyTorch, make sure ROCm shared libraries are on your environment first:

```bash
# bash/zsh
source scripts/rocm_env.sh

# fish
source scripts/rocm_env.fish
```

Run against a checkpoint (for Phase 3 micro-run, `latest.pth` works):

```bash
python3 scripts/phase4_monitor.py \
  --checkpoint data/runs/phase3_micro_run/<RUN_ID>/checkpoints/latest.pth \
  --split-manifest data/processed/_splits/val10_seed0.json \
  --split val
```

Outputs are written to a timestamped directory under:
- `data/monitor/phase4/<YYYYMMDD_HHMMSS>/`

Tip: if you use `scripts/phase2_setup_data_root.sh`, `data/monitor` will be symlinked to your NAS so artifacts don’t land on local disk.

Artifacts:
- `attention_heatmap.png`: a stable proxy heatmap derived from patch-token magnitudes
- `metrics.json`: includes embedding dispersion (std) and provenance/config

## 4) Automation

This monitor is designed to be safe to run in parallel with training (read-only).
A simple approach is a daily cron entry that runs `phase4_monitor.py` on `latest.pth`.
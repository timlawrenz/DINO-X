#!/usr/bin/env bash
# Ablation scan: 5-dataset bs256 50K checkpoint → close the AUROC gap
#
# Baseline: LoRA r=8, 64px crops, lung HU window → AUROC 0.684
# Target:   4-dataset model → AUROC 0.710
#
# Experiments 2-4 (experiment 1 = 100K pretraining, needs Strix Halo)
#
# Usage:
#   ./scripts/ablation_5dataset_bs256.sh          # run all
#   ./scripts/ablation_5dataset_bs256.sh rank      # rank ablation only
#   ./scripts/ablation_5dataset_bs256.sh unfreeze  # partial unfreeze only
#   ./scripts/ablation_5dataset_bs256.sh crop128   # 128px crop only
#   ./scripts/ablation_5dataset_bs256.sh baseline  # reproduce baseline (seed=42)
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
BACKBONE="runs/20260423_171906_5dataset-phase3-small-bs256/checkpoint_final_00050000.pth"
TRAIN_CSV="/mnt/nas-ai-models/training-data/dino-x/lidc-idri/labels/malignancy_train.csv"
VAL_CSV="/mnt/nas-ai-models/training-data/dino-x/lidc-idri/labels/malignancy_val.csv"
ADAPTER_BASE="adapters/ablation-5dataset-bs256"

SEED=42
COMMON_ARGS=(
    --train-csv "$TRAIN_CSV"
    --val-csv "$VAL_CSV"
    --input-format hu16_png
    --window-level -30 --window-width 120
    --task classification --num-classes 2
    --epochs 50 --batch-size 32 --patience 10
    --es-metric auroc --warmup-epochs 3
    --seed "$SEED"
)

EXPERIMENT="${1:-all}"

run_experiment() {
    local name="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Experiment: $name"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    PYTHONUNBUFFERED=1 python scripts/finetune_lora.py "$@"
    echo ""
    echo "  → $name complete"
    echo ""
}

# ── Baseline (seeded, for fair comparison) ────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "baseline" ]]; then
    run_experiment "Baseline (r=8, seed=42)" \
        --backbone "$BACKBONE" \
        "${COMMON_ARGS[@]}" \
        --rank 8 --alpha 16 --lr 5e-4 \
        --output "${ADAPTER_BASE}-baseline-r8-seed42"
fi

# ── Experiment 2a: LoRA rank 16 ──────────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "rank" ]]; then
    run_experiment "LoRA rank=16" \
        --backbone "$BACKBONE" \
        "${COMMON_ARGS[@]}" \
        --rank 16 --alpha 32 --lr 5e-4 \
        --output "${ADAPTER_BASE}-r16-seed42"

    # ── Experiment 2b: LoRA rank 32 ──────────────────────────────────
    run_experiment "LoRA rank=32" \
        --backbone "$BACKBONE" \
        "${COMMON_ARGS[@]}" \
        --rank 32 --alpha 64 --lr 5e-4 \
        --output "${ADAPTER_BASE}-r32-seed42"
fi

# ── Experiment 3: Partial unfreezing (1 block) ──────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "unfreeze" ]]; then
    run_experiment "Unfreeze last 1 block (backbone_lr=1e-5)" \
        --backbone "$BACKBONE" \
        "${COMMON_ARGS[@]}" \
        --rank 8 --alpha 16 --lr 5e-4 \
        --unfreeze-blocks 1 --backbone-lr 1e-5 \
        --output "${ADAPTER_BASE}-unfreeze1-seed42"
fi

# ── Experiment 4: 128px crops ────────────────────────────────────────
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "crop128" ]]; then
    # Create temporary CSVs with 128px crop paths
    TMPDIR_128=$(mktemp -d)
    trap 'rm -rf "$TMPDIR_128"' EXIT

    sed 's|nodule-crops/|nodule-crops-128/|g' "$TRAIN_CSV" > "$TMPDIR_128/train.csv"
    sed 's|nodule-crops/|nodule-crops-128/|g' "$VAL_CSV" > "$TMPDIR_128/val.csv"

    run_experiment "128px crops" \
        --backbone "$BACKBONE" \
        --train-csv "$TMPDIR_128/train.csv" \
        --val-csv "$TMPDIR_128/val.csv" \
        --input-format hu16_png \
        --window-level -30 --window-width 120 \
        --task classification --num-classes 2 \
        --epochs 50 --batch-size 32 --patience 10 \
        --es-metric auroc --warmup-epochs 3 \
        --seed "$SEED" \
        --rank 8 --alpha 16 --lr 5e-4 \
        --output "${ADAPTER_BASE}-crop128-seed42"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All requested experiments complete."
echo "  Adapter directories: ${ADAPTER_BASE}-*"
echo "  Check finetune_config.json in each for results."
echo "════════════════════════════════════════════════════════════"

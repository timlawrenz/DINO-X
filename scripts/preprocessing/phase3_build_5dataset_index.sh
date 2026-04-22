#!/usr/bin/env bash
# Phase 3: Build 5-dataset combined index with CQ500
#
# Prerequisites:
#   1. CQ500 downloaded and preprocessed:
#      bash scripts/preprocessing/download_cq500.sh
#
#   2. Per-dataset index CSVs with spacing columns exist at:
#      - LIDC:    on NAS at lidc-idri/processed-hu16/_index/index_with_spacing.csv
#      - Pancreas: on NAS at pancreas-ct/processed-hu16/_index/index.csv
#      - MSD Colon: on NAS at msd-colon/index.csv
#      - MSD Hepatic: on NAS at msd-hepatic-vessel/index.csv
#      - CQ500:   data/processed/cq500/index.csv (generated in step 1)
#
# This script:
#   1. Combines all 5 datasets into one CSV
#   2. Applies temperature-scaled sampling (T=2.0)
#   3. Generates a train/val split manifest
#
set -euo pipefail

NAS="/mnt/nas-ai-models/training-data/dino-x"
OUT_DIR="data/mvp"
mkdir -p "$OUT_DIR"

# ── Step 1: Locate per-dataset indices ──────────────────────────────
LIDC_IDX="${NAS}/lidc-idri/processed-hu16/_index/index_with_spacing.csv"
PANCREAS_IDX="${NAS}/pancreas-ct/processed-hu16/_index/index.csv"
MSD_COLON_IDX="${NAS}/msd-colon/index.csv"
MSD_HEPATIC_IDX="${NAS}/msd-hepatic-vessel/index.csv"
CQ500_IDX="data/processed/cq500/index.csv"

echo "=== Checking per-dataset indices ==="
for f in "$LIDC_IDX" "$PANCREAS_IDX" "$MSD_COLON_IDX" "$MSD_HEPATIC_IDX"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing index: $f"
        exit 1
    fi
    count=$(wc -l < "$f")
    echo "  OK: $f ($count rows)"
done

if [ ! -f "$CQ500_IDX" ]; then
    echo ""
    echo "ERROR: CQ500 index not found at: $CQ500_IDX"
    echo ""
    echo "To preprocess CQ500, run:"
    echo "  python scripts/preprocessing/phase2_preprocess_lidc_idri.py \\"
    echo "    --dicom-root data/raw/cq500 \\"
    echo "    --out-root data/processed \\"
    echo "    --dataset-name cq500"
    exit 1
fi
count=$(wc -l < "$CQ500_IDX")
echo "  OK: $CQ500_IDX ($count rows)"

# ── Step 2: Combine 5-dataset index (raw, no oversampling) ─────────
echo ""
echo "=== Combining 5 datasets ==="
python scripts/preprocessing/mvp_combine_indices.py \
    --inputs "lidc-idri:${LIDC_IDX}" \
    --inputs "pancreas-ct:${PANCREAS_IDX}" \
    --inputs "msd-colon:${MSD_COLON_IDX}" \
    --inputs "msd-hepatic-vessel:${MSD_HEPATIC_IDX}" \
    --inputs "cq500:${CQ500_IDX}" \
    --out "${OUT_DIR}/combined_5dataset.csv"

# ── Step 3: Apply temperature-scaled sampling ──────────────────────
echo ""
echo "=== Applying temperature-scaled sampling (T=2.0) ==="
python -c "
import csv, math, random, sys
from collections import Counter

random.seed(42)
inpath = '${OUT_DIR}/combined_5dataset.csv'
outpath = '${OUT_DIR}/combined_5dataset_t2.csv'

with open(inpath) as f:
    reader = csv.DictReader(f)
    rows_by_ds = {}
    for row in reader:
        ds = row['dataset']
        rows_by_ds.setdefault(ds, []).append(row)

sizes = {ds: len(rows) for ds, rows in rows_by_ds.items()}
T = 2.0
raw_weights = {ds: math.pow(n, 1.0/T) for ds, n in sizes.items()}
total_w = sum(raw_weights.values())
target_total = sum(sizes.values())
target_per_ds = {ds: int(round(w / total_w * target_total)) for ds, w in raw_weights.items()}

print(f'Total raw slices: {sum(sizes.values())}')
print(f'Temperature-scaled targets:')
all_rows = []
for ds in sorted(sizes.keys()):
    target = target_per_ds[ds]
    src = rows_by_ds[ds]
    if target <= len(src):
        sampled = random.sample(src, target)
    else:
        sampled = list(src) + random.choices(src, k=target - len(src))
    print(f'  {ds}: {len(src)} -> {len(sampled)} (factor: {len(sampled)/len(src):.2f}x)')
    all_rows.extend(sampled)

random.shuffle(all_rows)
fieldnames = ['dataset','png_path','series_dir','slice_index','encoding','spacing_x','spacing_y','spacing_z']
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(all_rows)

print(f'\nWrote {len(all_rows)} rows to {outpath}')
"

# ── Step 4: Generate train/val split manifest ──────────────────────
echo ""
echo "=== Generating train/val split manifest ==="
python scripts/preprocessing/phase4_make_split_manifest.py \
    --index-csv "${OUT_DIR}/combined_5dataset_t2.csv" \
    --seed 42 \
    --val-frac 0.10 \
    --out "${OUT_DIR}/split_manifest_5dataset.json"

echo ""
echo "=== Phase 3 data ready ==="
echo "Combined index: ${OUT_DIR}/combined_5dataset_t2.csv"
echo "Split manifest: ${OUT_DIR}/split_manifest_5dataset.json"
echo ""
echo "To launch Phase 3 training:"
echo "  PYTHONUNBUFFERED=1 python scripts/phase5_big_run.py \\"
echo "    --index-csv ${OUT_DIR}/combined_5dataset_t2.csv \\"
echo "    --split-manifest ${OUT_DIR}/split_manifest_5dataset.json \\"
echo "    --run-dir runs/ \\"
echo "    --run-suffix 5dataset-phase3 \\"
echo "    --max-steps 50000 \\"
echo "    --warmup-steps 2500 \\"
echo "    --lr 2e-4 \\"
echo "    --batch-size 32 \\"
echo "    --accumulation-steps 4 \\"
echo "    --ckpt-every 2500 \\"
echo "    --ckpt-keep-last 20 \\"
echo "    --scale-aware \\"
echo "    --koleo-weight 0.1 \\"
echo "    --crop-scale-min 0.3 \\"
echo "    --z-stride 3 \\"
echo "    --diverse-batches \\"
echo "    --amp \\"
echo "    --amp-dtype bfloat16"

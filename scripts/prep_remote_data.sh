#!/usr/bin/env bash
# scripts/prep_remote_data.sh — Prepare MVP training data on a remote instance
#
# Downloads raw DICOM from TCIA, preprocesses to 16-bit HU PNG, creates the
# combined index + train/val split, and uploads the result to HuggingFace.
#
# Run on any Linux box with fast internet (e.g., Vast.ai CPU instance).
# The output is a single tar.gz on HuggingFace that training instances
# can pull at datacenter speed.
#
# Output structure (inside tar):
#   data/processed/
#     lidc-idri/         16-bit HU PNG slices (100 series, ~24K slices)
#     pancreas-ct/       16-bit HU PNG slices (~80 series, ~18K slices)
#     _index/            per-dataset index CSVs
#     combined-mvp/
#       index.csv        combined index with spacing columns
#       split_manifest.json   train/val split (90/10 by series)
#
# Prerequisites:
#   - HF_TOKEN environment variable set
#   - 50GB+ free disk space
#   - Python 3.10+ with pip
#
# Usage:
#   git clone https://github.com/timlawrenz/DINO-X.git && cd DINO-X
#   export HF_TOKEN=hf_...
#   bash scripts/prep_remote_data.sh
#
# Resume: re-running skips already-downloaded series and already-processed PNGs.

set -euo pipefail

# ── Configuration ──
SEED=42
LIDC_SERIES_COUNT=100
TCIA_COLLECTION_LIDC="LIDC-IDRI"
TCIA_COLLECTION_PANCREAS="Pancreas-CT"
HF_REPO="${HF_REPO:-timlawrenz/dinox-mvp-data}"
TAR_NAME="mvp-processed.tar.gz"
CACHE_DIR=""  # set below after REPO_ROOT

# ── Validate ──
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable must be set" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
CACHE_DIR="$REPO_ROOT/.cache/prep"
mkdir -p "$CACHE_DIR"
echo "repo_root=$(pwd)"
echo "hf_repo=$HF_REPO"
echo "seed=$SEED"
echo "lidc_series_count=$LIDC_SERIES_COUNT"

# ── Step 1: Install dependencies ──
echo ""
echo "=== Step 1/9: Install Python dependencies ==="
pip install --quiet numpy pillow pydicom huggingface-hub 2>&1 | tail -3

# ── Step 2: List LIDC-IDRI series from TCIA ──
echo ""
echo "=== Step 2/9: Query TCIA for LIDC-IDRI series ==="
mkdir -p data/raw

if [ -f data/raw/lidc_mvp_uids.txt ]; then
    echo "skip=true reason=lidc_mvp_uids.txt_exists"
else
    python scripts/preprocessing/phase2_tcia_download.py list-series \
        --collection "$TCIA_COLLECTION_LIDC" \
        --modality CT \
        --out data/raw/lidc_all_uids.txt

    python3 -c "
import random
uids = sorted(l.strip() for l in open('data/raw/lidc_all_uids.txt') if l.strip())
rng = random.Random($SEED)
selected = sorted(rng.sample(uids, min($LIDC_SERIES_COUNT, len(uids))))
with open('data/raw/lidc_mvp_uids.txt', 'w') as f:
    f.write('\n'.join(selected) + '\n')
print(f'selected={len(selected)} total={len(uids)}')
"
fi

# ── Step 3: Download LIDC-IDRI (100 series) ──
echo ""
echo "=== Step 3/9: Download LIDC-IDRI ($LIDC_SERIES_COUNT series) ==="
python scripts/preprocessing/phase2_tcia_download.py download-series \
    --collection "$TCIA_COLLECTION_LIDC" \
    --uids data/raw/lidc_mvp_uids.txt \
    --out-root data/raw/lidc-idri

# ── Step 4: Download Pancreas-CT (all volumes) ──
echo ""
echo "=== Step 4/9: Download Pancreas-CT (all volumes) ==="
python scripts/preprocessing/phase2_tcia_download.py download-collection \
    --collection "$TCIA_COLLECTION_PANCREAS" \
    --out-root data/raw/pancreas-ct \
    --out-uids data/raw/pancreas_all_uids.txt

# ── Step 5: Preprocess LIDC-IDRI ──
echo ""
echo "=== Step 5/9: Preprocess LIDC-IDRI (DICOM → 16-bit PNG) ==="
python scripts/preprocessing/phase2_preprocess_lidc_idri.py \
    --dicom-root data/raw/lidc-idri \
    --out-root data/processed \
    --dataset-name lidc-idri
cp data/processed/_index/index.csv data/processed/_index/lidc_index.csv
echo "lidc_index=$(wc -l < data/processed/_index/lidc_index.csv) lines"

# ── Step 6: Preprocess Pancreas-CT ──
echo ""
echo "=== Step 6/9: Preprocess Pancreas-CT (DICOM → 16-bit PNG) ==="
python scripts/preprocessing/phase2_preprocess_lidc_idri.py \
    --dicom-root data/raw/pancreas-ct \
    --out-root data/processed \
    --dataset-name pancreas-ct
cp data/processed/_index/index.csv data/processed/_index/pancreas_index.csv
echo "pancreas_index=$(wc -l < data/processed/_index/pancreas_index.csv) lines"

# ── Step 7: Combine indices ──
echo ""
echo "=== Step 7/9: Combine dataset indices ==="
python scripts/preprocessing/mvp_combine_indices.py \
    --inputs lidc-idri:data/processed/_index/lidc_index.csv \
    --inputs pancreas-ct:data/processed/_index/pancreas_index.csv \
    --out data/processed/combined-mvp/index.csv

# ── Step 8: Create train/val split ──
echo ""
echo "=== Step 8/9: Create train/val split manifest ==="
python3 << 'SPLIT_EOF'
import csv, json, random, os

index_path = "data/processed/combined-mvp/index.csv"
out_path = "data/processed/combined-mvp/split_manifest.json"

rows = []
with open(index_path, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

all_series = sorted({r["series_dir"] for r in rows})
rng = random.Random(42)
rng.shuffle(all_series)

n_val = max(1, len(all_series) // 10)  # 10% for validation
val_series = sorted(all_series[:n_val])
train_series = sorted(all_series[n_val:])

manifest = {
    "train": {"series_dir": train_series},
    "val": {"series_dir": val_series},
}
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

# Count slices per split
val_set = set(val_series)
n_train_slices = sum(1 for r in rows if r["series_dir"] not in val_set)
n_val_slices = sum(1 for r in rows if r["series_dir"] in val_set)

print(f"train_series={len(train_series)} val_series={len(val_series)}")
print(f"train_slices={n_train_slices} val_slices={n_val_slices}")
print(f"out={out_path}")
SPLIT_EOF

# ── Step 9: Tar and upload to HuggingFace ──
echo ""
echo "=== Step 9/9: Tar and upload to HuggingFace ==="

echo "Creating archive..."
tar -czf "$CACHE_DIR/$TAR_NAME" \
    data/processed/lidc-idri/ \
    data/processed/pancreas-ct/ \
    data/processed/combined-mvp/ \
    data/processed/_index/

echo "tar_size=$(du -sh "$CACHE_DIR/$TAR_NAME" | cut -f1)"

echo "Uploading to $HF_REPO..."
huggingface-cli upload "$HF_REPO" \
    "$CACHE_DIR/$TAR_NAME" "$TAR_NAME" \
    --repo-type dataset

rm -f "$CACHE_DIR/$TAR_NAME"

# ── Summary ──
echo ""
echo "========================================="
echo "  MVP Data Preparation Complete"
echo "========================================="
du -sh data/processed/lidc-idri/ data/processed/pancreas-ct/
echo ""
wc -l data/processed/combined-mvp/index.csv
echo ""
echo "Uploaded to: https://huggingface.co/datasets/$HF_REPO"
echo ""
echo "Training instances pull data with:"
echo "  bash scripts/fetch_hf_data.sh"

#!/usr/bin/env bash
# Download and preprocess CQ500 head CT dataset from Kaggle.
#
# Prerequisites:
#   1. Kaggle API credentials at ~/.kaggle/kaggle.json
#      Get your token from: https://www.kaggle.com/settings → API → Create New Token
#      Then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#
#   2. Python venv with dependencies activated:
#      source .venv/bin/activate
#      pip install kaggle pydicom pillow numpy
#
# Dataset: https://www.kaggle.com/datasets/crawford/qureai-headct
# License: CC-BY-SA-4.0
# Size: ~26.6 GB (DICOM)
# Slices: ~193K across 491 head CTs
#
# Output:
#   data/cq500/raw/          — Raw DICOM files (NAS via symlink)
#   data/processed/cq500/    — 16-bit HU PNG slices (shared processed root)
#   data/processed/cq500/index.csv — Per-slice index with spacing metadata
#
set -euo pipefail

RAW_DIR="data/cq500/raw"
PROCESSED_DIR="data/processed"
DATASET_NAME="cq500"

echo "=== CQ500 Download & Preprocess Pipeline ==="
echo ""

# ── Step 1: Download from Kaggle ────────────────────────────────────
if [ -d "$RAW_DIR" ] && [ "$(find "$RAW_DIR" -name "*.dcm" -o -name "*.DCM" | head -1)" ]; then
    echo "Step 1: SKIP — DICOM files already exist in $RAW_DIR"
    echo "  $(find "$RAW_DIR" -type f | wc -l) files found"
else
    echo "Step 1: Downloading CQ500 from Kaggle (~26.6 GB)..."
    echo "  Source: kaggle.com/datasets/crawford/qureai-headct"
    echo ""

    # Check credentials
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "ERROR: Kaggle API credentials not found."
        echo ""
        echo "Setup instructions:"
        echo "  1. Go to https://www.kaggle.com/settings"
        echo "  2. Scroll to 'API' section → 'Create New Token'"
        echo "  3. Save the downloaded file:"
        echo "     mkdir -p ~/.kaggle"
        echo "     mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "     chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "Or download manually from:"
        echo "  https://www.kaggle.com/datasets/crawford/qureai-headct"
        echo "  Extract to: $RAW_DIR"
        exit 1
    fi

    mkdir -p "$RAW_DIR"

    # Download and unzip
    kaggle datasets download crawford/qureai-headct \
        --path "$RAW_DIR" \
        --unzip

    echo "  Download complete."
    echo "  $(find "$RAW_DIR" -type f | wc -l) files extracted"
fi

# ── Step 2: Validate DICOM data ────────────────────────────────────
echo ""
echo "Step 2: Validating DICOM data..."
N_DIRS=$(find "$RAW_DIR" -type d | wc -l)
N_FILES=$(find "$RAW_DIR" -type f | wc -l)
echo "  Directories: $N_DIRS"
echo "  Files: $N_FILES"

if [ "$N_FILES" -lt 1000 ]; then
    echo "WARNING: Expected ~193K files but only found $N_FILES"
    echo "  The download may be incomplete or the data may need further extraction."
    echo "  Check $RAW_DIR for .zip files that need unzipping."
    # Check for nested zips
    ZIPS=$(find "$RAW_DIR" -name "*.zip" | wc -l)
    if [ "$ZIPS" -gt 0 ]; then
        echo "  Found $ZIPS .zip files — extracting..."
        find "$RAW_DIR" -name "*.zip" -exec sh -c 'echo "  Extracting: {}"; unzip -q -o "{}" -d "$(dirname "{}")"' \;
        N_FILES=$(find "$RAW_DIR" -type f -not -name "*.zip" | wc -l)
        echo "  After extraction: $N_FILES files"
    fi
fi

# ── Step 3: Preprocess DICOM → 16-bit HU PNG ──────────────────────
echo ""
INDEX_FILE="$PROCESSED_DIR/$DATASET_NAME/index.csv"
if [ -f "$INDEX_FILE" ]; then
    EXISTING=$(wc -l < "$INDEX_FILE")
    echo "Step 3: Index already exists at $INDEX_FILE ($EXISTING rows)"
    echo "  Use --force-reprocess to regenerate."
    echo "  Skipping preprocessing."
else
    echo "Step 3: Preprocessing DICOM → 16-bit HU PNG..."
    echo "  This will take 15-30 minutes depending on disk speed."
    echo ""

    PYTHONUNBUFFERED=1 python scripts/preprocessing/phase2_preprocess_lidc_idri.py \
        --dicom-root "$RAW_DIR" \
        --out-root "$PROCESSED_DIR" \
        --dataset-name "$DATASET_NAME"

    if [ -f "$INDEX_FILE" ]; then
        N_SLICES=$(tail -n +2 "$INDEX_FILE" | wc -l)
        N_SERIES=$(tail -n +2 "$INDEX_FILE" | cut -d',' -f2 | sort -u | wc -l)
        echo ""
        echo "  Preprocessing complete:"
        echo "    Slices: $N_SLICES"
        echo "    Series: $N_SERIES"
        echo "    Index: $INDEX_FILE"
    else
        echo "ERROR: Preprocessing did not produce index at $INDEX_FILE"
        exit 1
    fi
fi

# ── Step 4: Validate spacing metadata ──────────────────────────────
echo ""
echo "Step 4: Validating spacing metadata..."
python3 -c "
import csv, sys
from collections import Counter

with open('$INDEX_FILE') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

n = len(rows)
series = set(r['series_dir'] for r in rows)

# Check spacing columns exist and are not default 1.0
has_spacing = all(k in rows[0] for k in ('spacing_x', 'spacing_y', 'spacing_z'))
if not has_spacing:
    print('ERROR: Index missing spacing columns!')
    sys.exit(1)

sx = [float(r['spacing_x']) for r in rows]
sy = [float(r['spacing_y']) for r in rows]
sz = [float(r['spacing_z']) for r in rows]

default_count = sum(1 for x, y, z in zip(sx, sy, sz) if x == 1.0 and y == 1.0 and z == 1.0)

print(f'  Slices: {n}')
print(f'  Series: {len(series)}')
print(f'  Spacing X range: [{min(sx):.3f}, {max(sx):.3f}]')
print(f'  Spacing Y range: [{min(sy):.3f}, {max(sy):.3f}]')
print(f'  Spacing Z range: [{min(sz):.3f}, {max(sz):.3f}]')
if default_count > 0:
    pct = default_count / n * 100
    print(f'  WARNING: {default_count}/{n} ({pct:.1f}%) slices have default spacing (1.0, 1.0, 1.0)')
else:
    print(f'  All slices have non-default spacing metadata ✓')
"

echo ""
echo "=== CQ500 ready for Phase 3 ==="
echo ""
echo "Next: Run the 5-dataset index builder:"
echo "  bash scripts/preprocessing/phase3_build_5dataset_index.sh"

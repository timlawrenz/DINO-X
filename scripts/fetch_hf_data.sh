#!/usr/bin/env bash
# scripts/fetch_hf_data.sh — Download preprocessed MVP data from HuggingFace
#
# Run on training instances (or locally) to pull the preprocessed dataset.
# After running, data/processed/ will contain the combined index and PNG slices
# ready for phase5_big_run.py.
#
# Prerequisites:
#   - HF_TOKEN environment variable set
#   - huggingface-hub installed (pip install huggingface-hub)
#
# Usage (from repo root):
#   export HF_TOKEN=hf_...
#   bash scripts/fetch_hf_data.sh

set -euo pipefail

HF_REPO="${HF_REPO:-timlawrenz/dinox-mvp-data}"
TAR_BASE="mvp-processed.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CACHE_DIR="$REPO_ROOT/.cache/hf-data"
COMPLETE_MARKER="data/processed/.fetch_complete"

# Skip if data already fully extracted
if [ -f "$COMPLETE_MARKER" ] && [ -f data/processed/combined-mvp/index.csv ]; then
    N=$(wc -l < data/processed/combined-mvp/index.csv)
    echo "skip=true reason=data_exists lines=$N"
    echo "To force re-download, remove data/processed/ and $COMPLETE_MARKER"
    exit 0
fi

mkdir -p "$CACHE_DIR"

# Download split parts + metadata from HuggingFace
echo "Downloading data from $HF_REPO ..."
for PART in part_aa part_ab part_ac part_ad part_ae; do
    FILE="${TAR_BASE}.${PART}"
    if [ -f "$CACHE_DIR/$FILE" ]; then
        echo "skip=$FILE reason=already_downloaded"
        continue
    fi
    echo "download=$FILE"
    hf download "$HF_REPO" "$FILE" \
        --repo-type dataset \
        --local-dir "$CACHE_DIR" \
        --token "${HF_TOKEN:-}"
done

# Also download index + split manifest (small files, useful standalone)
for META in index.csv split_manifest.json; do
    hf download "$HF_REPO" "$META" \
        --repo-type dataset \
        --local-dir "$CACHE_DIR" \
        --token "${HF_TOKEN:-}" 2>/dev/null || true
done

# Reassemble split tar and extract
echo "Reassembling tar from parts..."
cat "$CACHE_DIR/${TAR_BASE}".part_* > "$CACHE_DIR/$TAR_BASE"

echo "Extracting to $REPO_ROOT ..."
tar -xzf "$CACHE_DIR/$TAR_BASE" -C "$REPO_ROOT"

# Mark extraction complete
touch "$COMPLETE_MARKER"

# Copy metadata into combined-mvp for easy access
mkdir -p data/processed/combined-mvp
for META in index.csv split_manifest.json; do
    [ -f "$CACHE_DIR/$META" ] && cp "$CACHE_DIR/$META" "data/processed/combined-mvp/$META"
done

# Clean up cache
rm -f "$CACHE_DIR/${TAR_BASE}"*
rmdir "$CACHE_DIR" 2>/dev/null || true

echo "ok=true"
echo "index=$(wc -l < data/processed/combined-mvp/index.csv) lines"
du -sh data/processed/lidc-idri/ data/processed/pancreas-ct/

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
TAR_NAME="mvp-processed.tar.gz"

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

echo "Downloading $TAR_NAME from $HF_REPO ..."
huggingface-cli download "$HF_REPO" "$TAR_NAME" \
    --repo-type dataset \
    --local-dir "$CACHE_DIR"

echo "Extracting to $REPO_ROOT ..."
tar -xzf "$CACHE_DIR/$TAR_NAME" -C "$REPO_ROOT"

# Mark extraction complete
touch "$COMPLETE_MARKER"

rm -f "$CACHE_DIR/$TAR_NAME"
rmdir "$CACHE_DIR" 2>/dev/null || true

echo "ok=true"
echo "index=$(wc -l < data/processed/combined-mvp/index.csv) lines"
du -sh data/processed/lidc-idri/ data/processed/pancreas-ct/

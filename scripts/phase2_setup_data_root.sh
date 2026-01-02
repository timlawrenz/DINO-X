#!/usr/bin/env bash
set -euo pipefail

# Phase 2 helper: keep raw/derived data off-repo, but make local paths stable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

DINOX_DATA_ROOT="${DINOX_DATA_ROOT:-/mnt/nas-ai-models/training-data/dino-x}"
LIDC_ROOT="$DINOX_DATA_ROOT/lidc-idri"

RAW_TARGET="$LIDC_ROOT/raw"
PROC_TARGET="$LIDC_ROOT/processed-2p5d-rgb"

mkdir -p "$DATA_DIR"
mkdir -p "$RAW_TARGET" "$PROC_TARGET"

ln -sfn "$RAW_TARGET" "$DATA_DIR/raw"
ln -sfn "$PROC_TARGET" "$DATA_DIR/processed"

echo "ok=true"
echo "data/raw -> $(readlink -f "$DATA_DIR/raw")"
echo "data/processed -> $(readlink -f "$DATA_DIR/processed")"

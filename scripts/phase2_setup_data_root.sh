#!/usr/bin/env bash
set -euo pipefail

# Phase 2 helper: keep raw/derived data off-repo, but make local paths stable.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

DINOX_DATA_ROOT="${DINOX_DATA_ROOT:-/mnt/nas-ai-models/training-data/dino-x}"
LIDC_ROOT="$DINOX_DATA_ROOT/lidc-idri"

RAW_TARGET="$LIDC_ROOT/raw"
PROC_TARGET="$LIDC_ROOT/processed-hu16"
RUNS_TARGET="$DINOX_DATA_ROOT/runs"
MONITOR_TARGET="$DINOX_DATA_ROOT/monitor"

mkdir -p "$DATA_DIR"
mkdir -p "$RAW_TARGET" "$PROC_TARGET" "$RUNS_TARGET" "$MONITOR_TARGET"

# If a local monitor directory already exists (not a symlink), migrate it to the NAS
# and replace it with a symlink.
#
# NOTE: If a previous run created `data/monitor/monitor -> $MONITOR_TARGET` inside that
# directory, moving the directory under $MONITOR_TARGET would create a recursive loop.
if [[ -e "$DATA_DIR/monitor" && ! -L "$DATA_DIR/monitor" ]]; then
  # Remove known self-referential nested symlink if present.
  if [[ -L "$DATA_DIR/monitor/monitor" ]] && [[ "$(readlink -f "$DATA_DIR/monitor/monitor")" == "$(readlink -f "$MONITOR_TARGET")" ]]; then
    rm "$DATA_DIR/monitor/monitor"
  fi

  TS="$(date -u +%Y%m%d_%H%M%S)"
  MIGRATED="$MONITOR_TARGET/migrated_local_${TS}"
  echo "warn=true msg=monitor_dir_exists_migrating from=$DATA_DIR/monitor to=$MIGRATED" >&2
  mv "$DATA_DIR/monitor" "$MIGRATED"
fi

ln -sfn "$RAW_TARGET" "$DATA_DIR/raw"
ln -sfn "$PROC_TARGET" "$DATA_DIR/processed"
ln -sfn "$RUNS_TARGET" "$DATA_DIR/runs"
ln -sfn "$MONITOR_TARGET" "$DATA_DIR/monitor"

echo "ok=true"
echo "data/raw -> $(readlink -f "$DATA_DIR/raw")"
echo "data/processed -> $(readlink -f "$DATA_DIR/processed")"
echo "data/runs -> $(readlink -f "$DATA_DIR/runs")"
echo "data/monitor -> $(readlink -f "$DATA_DIR/monitor")"

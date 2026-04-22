#!/usr/bin/env bash
#
# setup_comfyui_rocm_8060s.sh
#
# IMPORTANT:
#   - You MUST have ROCm installed already (drivers + runtime).
#   - You MUST download the ROCm-enabled PyTorch wheels that match your
#     ROCm version and Python version from AMD's repo before running the
#     install step below.
#   - Review and edit the CONFIG section before executing.

set -euo pipefail

########################################
# CONFIG - EDIT THESE FOR YOUR SYSTEM #
########################################

# Python interpreter to use for the virtual environment.
# This script is configured for ROCm 7.1.1 wheels built for Python 3.13 (cp313).
PYTHON_BIN="python3.13"  # change if your Python 3.13 binary has a different name

# Name of the virtual environment directory inside this ComfyUI repo.
VENV_DIR=".venv"

# Directory where you downloaded ROCm PyTorch wheels (torch/torchvision/torchaudio/triton).
# In this setup we assume you used the local ./rocm-wheels directory in this repo.
ROCM_WHEEL_DIR="$(pwd)/rocm-wheels"  # adjust if your wheels are elsewhere

# Individual wheel paths (edit to the exact filenames you downloaded).
# These defaults match ROCm 7.1.1 cp313 wheels as of December 2025.
TORCH_WHL="${ROCM_WHEEL_DIR}/torch-2.9.1+rocm7.1.1.lw.git351ff442-cp313-cp313-linux_x86_64.whl"
TORCHVISION_WHL="${ROCM_WHEEL_DIR}/torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp313-cp313-linux_x86_64.whl"
TORCHAUDIO_WHL="${ROCM_WHEEL_DIR}/torchaudio-2.9.0+rocm7.1.1.gite3c6ee2b-cp313-cp313-linux_x86_64.whl"
TRITON_WHL="${ROCM_WHEEL_DIR}/triton-3.5.1+rocm7.1.1.gita272dfa8-cp313-cp313-linux_x86_64.whl"

########################################
# HELPER FUNCTIONS                     #
########################################

echo_step() {
  echo
  echo "=== $1 ==="
}

echo_info() {
  echo "[INFO] $1"
}

echo_err() {
  echo "[ERROR] $1" >&2
}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo_err "Required file not found: $f"
    echo_err "Edit the CONFIG section at the top of this script to point to your actual ROCm wheels."
    exit 1
  fi
}

########################################
# 1. CREATE PYTHON VENV                #
########################################

echo_step "Step 1: Create and activate Python virtual environment"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo_err "Python binary '$PYTHON_BIN' not found. Install it or set PYTHON_BIN to an existing interpreter."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo_info "Creating venv at: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo_info "Reusing existing venv at: $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo_info "Using Python: $(command -v python)"
python --version

########################################
# 2. INSTALL ROCM PYTORCH WHEELS      #
########################################

echo_step "Step 2: Install ROCm-enabled PyTorch wheels"

require_file "$TORCH_WHL"
require_file "$TORCHVISION_WHL"
require_file "$TORCHAUDIO_WHL"
require_file "$TRITON_WHL"

echo_info "Uninstalling any existing torch/vision/audio/triton from this venv (if present)"
python -m pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm 2>/dev/null || true

echo_info "Installing ROCm wheels from: $ROCM_WHEEL_DIR"
python -m pip install \
  "$TORCH_WHL" \
  "$TORCHVISION_WHL" \
  "$TORCHAUDIO_WHL" \
  "$TRITON_WHL"

########################################
# 3. INSTALL REQUIREMENTS             #
########################################

echo_step "Step 3: Install Python requirements (without overriding ROCm torch)"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

########################################
# 4. SANITY CHECK TORCH + ROCM        #
########################################

echo_step "Step 4: Sanity check that PyTorch sees ROCm and your Radeon 8060S"

# Ensure ROCm libraries are visible to the venv before importing torch.
if [[ -d "/opt/rocm" ]]; then
  export PATH="/opt/rocm/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
  echo_info "[Step 4] Updated PATH and LD_LIBRARY_PATH to include /opt/rocm for sanity check"
fi

python - << 'EOF'
import torch
print("torch version:", torch.__version__)
print("HIP version:", torch.version.hip)
print("cuda_is_available (ROCm):", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
EOF

########################################
# 5. PHASE 1 ATTENTION SMOKE TEST      #
########################################

echo_step "Step 5: Phase 1 attention smoke test (512x512)"

# Enable experimental mem-efficient SDPA path on ROCm (recommended on Strix Halo).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

python scripts/phase1_validate_attention.py --size 512 --device cuda --dtype fp16

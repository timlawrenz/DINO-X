#!/usr/bin/env bash
# Source this to make ROCm userland libraries visible to Python wheels.
# Usage:
#   source scripts/rocm_env.sh

if [ -d /opt/rocm ]; then
  export PATH="/opt/rocm/bin:${PATH}"
  export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
fi

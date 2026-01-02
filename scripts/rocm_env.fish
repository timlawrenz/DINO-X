# Source this to make ROCm userland libraries visible to Python wheels.
# Usage:
#   source scripts/rocm_env.fish

if test -d /opt/rocm
    fish_add_path /opt/rocm/bin
    if set -q LD_LIBRARY_PATH
        set -gx LD_LIBRARY_PATH /opt/rocm/lib /opt/rocm/lib64 $LD_LIBRARY_PATH
    else
        set -gx LD_LIBRARY_PATH /opt/rocm/lib /opt/rocm/lib64
    end
end

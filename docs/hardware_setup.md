# Phase 1: Platform Bootstrap (Hardware & Environment)

This document corresponds to **Phase 1** in `docs/roadmap.md`: proving the Strix Halo platform can run the core math stack (ROCm + attention kernels) without crashing.

## 1) Hardware assembly & thermals

- Install the Strix Halo system in the intended chassis (Framework/Desktop or equivalent).
- In BIOS/UEFI, set fan control to a sustained-load profile ("Server"/"Turbo"/equivalent).
- Confirm cooling is stable under sustained load before attempting multi-day runs.

## 2) OS & kernel setup

- Install Linux.
- **Kernel requirement:** 6.11+ (6.15+ preferred).

Quick checks:

```bash
uname -r
```

## 3) ROCm 7.1 installation (gfx1151)

Install ROCm **7.1** with **gfx1151** support and ensure core libraries are present (MIOpen, hipBLASLt, RCCL).

### Shell environment (required for ROCm wheels)

If you installed ROCm userland under `/opt/rocm`, ensure it is on your PATH and its libraries are discoverable.

Bash/zsh:

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
```

Fish:

```fish
fish_add_path /opt/rocm/bin
set -gx LD_LIBRARY_PATH /opt/rocm/lib /opt/rocm/lib64 $LD_LIBRARY_PATH
```

Alternatively, source the repo helpers:

```bash
source scripts/rocm_env.sh
```

```fish
source scripts/rocm_env.fish
```

### Validation commands

```bash
rocminfo
hipconfig --full
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

## 4) Flash Attention 2 (Triton/CK backend)

On this stack, FlashAttention-style kernels are provided through **PyTorch SDPA** on ROCm (AOTriton/Triton backend). There is no separate `flash-attn` Python package required for the Phase 1 smoke test.

Record versions (copy/paste output into your run log):

```bash
hipconfig --full | head -n 40
python -c "import torch; print('torch', torch.__version__); print('hip', torch.version.hip)"
python -c "import triton; print('triton', triton.__version__)"  # if installed
```

Enable the experimental mem-efficient SDPA path (recommended on Strix Halo):

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Then run the attention smoke test in the next section (this also serves as the FlashAttention/SDPA smoke test).

## 5) Phase 1 success check: attention smoke test

Run the repo-provided script:

```bash
# If torch+ROCm is installed, use the ROCm device (PyTorch reports it as "cuda"):
python scripts/phase1_validate_attention.py --size 512 --device cuda --dtype fp16
```

**Success criteria:** completes without crashes/hangs and prints `ok=true`.

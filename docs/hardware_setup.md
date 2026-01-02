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

Validation commands (names may vary by distro/package):

```bash
rocminfo
hipinfo
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4) Flash Attention 2 (Triton/CK backend)

Compile/install Flash Attention 2 for the ROCm stack you installed.

- Record the exact versions (ROCm, PyTorch, Triton/CK, FlashAttention) and build flags.
- Run a minimal attention forward-pass test after installation.

## 5) Phase 1 success check: attention smoke test

Run the repo-provided script:

```bash
python scripts/phase1_validate_attention.py --size 512 --device auto --dtype fp16
```

**Success criteria:** completes without crashes/hangs and prints `ok=true`.

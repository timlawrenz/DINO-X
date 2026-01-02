#!/usr/bin/env python3
"""Phase 1 validation: run a 512x512 dot-product attention workload.

Roadmap Phase 1 success criteria is simply: run an attention workload at 512x512 and
exit cleanly (no crashes/hangs).

This script uses PyTorch's scaled dot-product attention and will fall back to CPU
if a GPU device is not available.
"""

from __future__ import annotations

import argparse
import time


def _pick_device(torch, prefer: str) -> str:
    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        msg = str(e)
        if "libroctx64.so" in msg:
            raise SystemExit(
                "Failed to import torch due to missing ROCm runtime libraries.\n"
                "Try sourcing the repo helper for your shell before running:\n"
                "  bash/zsh: source scripts/rocm_env.sh\n"
                "  fish:     source scripts/rocm_env.fish\n"
                "Or export LD_LIBRARY_PATH to include /opt/rocm/lib and /opt/rocm/lib64.\n\n"
                f"Original error: {e}"
            )
        raise

    device = _pick_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"torch={torch.__version__}")
    print(f"device={device}")
    if device == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")

    n = args.size
    torch.manual_seed(0)

    # Shapes: (B, H, L, D)
    q = torch.randn((1, 1, n, n), device=device, dtype=dtype)
    k = torch.randn((1, 1, n, n), device=device, dtype=dtype)
    v = torch.randn((1, 1, n, n), device=device, dtype=dtype)

    t0 = time.time()
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    if not torch.isfinite(out).all():
        raise RuntimeError("Non-finite values detected in attention output")

    print(
        f"ok=true elapsed_s={t1 - t0:.6f} out_mean={out.mean().item():.6f} out_std={out.std().item():.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

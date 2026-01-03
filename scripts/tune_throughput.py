#!/usr/bin/env python3
"""Phase 4.5: Throughput tuning (Virtual Inflation).

Goal: stress-test the DataLoader + GPU step loop *as if* the dataset were huge,
without requiring the full 120GB download to finish.

This tool intentionally reuses Phase 3's dataset implementation (PngDataset)
via a dynamic import to avoid drift.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import os
import random
import resource
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(
        f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry. "
        "If you're using ROCm PyTorch, ensure ROCm libs are discoverable (e.g., `source scripts/rocm_env.*`)."
    )


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as ckpt
except Exception:  # pragma: no cover
    _need("torch")


def _load_phase3_module() -> Any:
    p = Path(__file__).resolve().parent / "phase3_micro_run.py"
    spec = importlib.util.spec_from_file_location("phase3_micro_run", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in (p.strip() for p in s.split(",")):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit(f"empty list: {s!r}")
    return out


def _parse_bool_list(s: str) -> list[bool]:
    out: list[bool] = []
    for part in (p.strip().lower() for p in s.split(",")):
        if not part:
            continue
        if part in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif part in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise SystemExit(f"invalid bool: {part!r} (from {s!r})")
    if not out:
        raise SystemExit(f"empty bool list: {s!r}")
    return out


class VirtualInflatedDataset(torch.utils.data.Dataset):
    """Repeats an underlying dataset K times without duplicating files on disk."""

    def __init__(self, base: torch.utils.data.Dataset, factor: int) -> None:
        if factor <= 0:
            raise ValueError("factor must be > 0")
        self.base = base
        self.factor = factor
        self.base_len = len(base)
        if self.base_len <= 0:
            raise ValueError("base dataset is empty")

    def __len__(self) -> int:  # pragma: no cover
        return self.base_len * self.factor

    def __getitem__(self, idx: int) -> Any:
        return self.base[idx % self.base_len]


class _TinyHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass(frozen=True)
class Combo:
    batch_size: int
    num_workers: int
    pin_memory: bool


def _torch_mem_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
            "max_reserved": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return None


def _cuda_mem_info() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return {"free": int(free_b), "total": int(total_b)}
    except Exception:
        return None


def _detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_sdp_backend(kind: str) -> None:
    # These APIs are under torch.backends.cuda even on ROCm builds.
    if not hasattr(torch.backends, "cuda"):
        return
    if kind == "auto":
        return
    if kind == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if kind == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    raise SystemExit(f"unknown --sdp-backend: {kind}")


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "hip out of memory" in msg


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--subset-size", type=int, default=1000)
    ap.add_argument("--subset-seed", type=int, default=0)

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--rw-level-min", type=float, default=-700.0)
    ap.add_argument("--rw-level-max", type=float, default=100.0)
    ap.add_argument("--rw-width-min", type=float, default=300.0)
    ap.add_argument("--rw-width-max", type=float, default=2000.0)

    ap.add_argument("--inflation-factor", type=int, default=1000)

    ap.add_argument("--batch-sizes", type=str, default="32,64,128,192,256,512")
    ap.add_argument("--num-workers", type=str, default="0,4,8,16,24,32")
    ap.add_argument("--pin-memory", type=str, default="true,false")

    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--bench-steps", type=int, default=20)
    ap.add_argument("--train-seed", type=int, default=0)

    ap.add_argument(
        "--sdp-backend",
        choices=["auto", "math", "mem_efficient"],
        default="auto",
        help="Force scaled_dot_product_attention backend (math is slowest but most stable)",
    )

    ap.add_argument(
        "--model",
        choices=["tiny", "dino"],
        default="tiny",
        help="Model used for the step loop: 'tiny' stresses the DataLoader; 'dino' is closer to Phase 3/5 student-teacher memory/compute.",
    )
    ap.add_argument("--vit-patch", type=int, default=16)
    ap.add_argument("--vit-dim", type=int, default=256)
    ap.add_argument("--vit-depth", type=int, default=2)
    ap.add_argument("--vit-heads", type=int, default=4)
    ap.add_argument("--vit-mlp-ratio", type=float, default=2.0)
    ap.add_argument("--dino-out-dim", type=int, default=8192)

    ap.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable activation checkpointing for the student (reduces memory, increases compute)",
    )

    ap.add_argument(
        "--amp",
        action="store_true",
        help="Enable autocast mixed precision (recommended for --model dino)",
    )
    ap.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Autocast dtype when --amp is enabled (bf16 recommended on newer ROCm).",
    )

    ap.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)",
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/runs/throughput_tuning") / time.strftime("%Y%m%d_%H%M%S"),
    )

    args = ap.parse_args()

    m = _load_phase3_module()

    # Snapshot the current index CSV once; background downloads may be changing.
    all_rows = m._load_index_rows(args.index_csv)

    rng = random.Random(args.subset_seed)
    rng.shuffle(all_rows)
    subset = all_rows[: min(args.subset_size, len(all_rows))]

    # Filter out rows whose primary png_path is missing at snapshot time.
    # (New files that appear later are intentionally ignored for reproducibility.)
    subset = [r for r in subset if Path(r.png_path).exists()]
    if not subset:
        raise SystemExit("No usable rows after filtering to existing png_path entries")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist reproducibility artifacts.
    _write_json(
        out_dir / "subset.json",
        [
            {
                "png_path": str(r.png_path),
                "series_dir": str(r.series_dir),
                "slice_index": int(r.slice_index),
                "encoding": str(r.encoding),
            }
            for r in subset
        ],
    )

    env = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "hip": getattr(getattr(torch, "version", None), "hip", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            env["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    _set_sdp_backend(args.sdp_backend)

    cfg = {
        "index_csv": str(args.index_csv),
        "subset_size": int(args.subset_size),
        "subset_seed": int(args.subset_seed),
        "img_size": int(args.img_size),
        "rw_level_min": float(args.rw_level_min),
        "rw_level_max": float(args.rw_level_max),
        "rw_width_min": float(args.rw_width_min),
        "rw_width_max": float(args.rw_width_max),
        "inflation_factor": int(args.inflation_factor),
        "batch_sizes": _parse_int_list(args.batch_sizes),
        "num_workers": _parse_int_list(args.num_workers),
        "pin_memory": _parse_bool_list(args.pin_memory),
        "warmup_steps": int(args.warmup_steps),
        "bench_steps": int(args.bench_steps),
        "train_seed": int(args.train_seed),
        "sdp_backend": str(args.sdp_backend),
        "model": str(args.model),
        "vit_patch": int(args.vit_patch),
        "vit_dim": int(args.vit_dim),
        "vit_depth": int(args.vit_depth),
        "vit_heads": int(args.vit_heads),
        "vit_mlp_ratio": float(args.vit_mlp_ratio),
        "dino_out_dim": int(args.dino_out_dim),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "grad_accum_steps": int(args.grad_accum_steps),
    }
    _write_json(out_dir / "config.json", {"env": env, "config": cfg})

    # Build dataset (reuse Phase 3 implementation) and apply virtual inflation.
    base_ds = m.PngDataset(
        subset,
        img_size=args.img_size,
        rw_level_min=args.rw_level_min,
        rw_level_max=args.rw_level_max,
        rw_width_min=args.rw_width_min,
        rw_width_max=args.rw_width_max,
    )
    ds = VirtualInflatedDataset(base_ds, factor=args.inflation_factor)

    device = _detect_device()

    # Small but non-trivial compute kernel to exercise forward/backward.
    # Note: Phase 3's PatchViT does not define forward(); it exposes forward_features().

    class _CheckpointedPatchViT(nn.Module):
        def __init__(
            self,
            *,
            img_size: int,
            patch: int,
            dim: int,
            depth: int,
            heads: int,
            mlp_ratio: float,
        ) -> None:
            super().__init__()
            assert img_size % patch == 0
            self.img_size = img_size
            self.patch = patch
            self.dim = dim

            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
            n_patches = (img_size // patch) * (img_size // patch)

            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))

            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=dim,
                        nhead=heads,
                        dim_feedforward=int(dim * mlp_ratio),
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(depth)
                ]
            )
            self.norm = nn.LayerNorm(dim)

            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos

            for layer in self.layers:
                x = ckpt.checkpoint(layer, x, use_reentrant=False)
            x = self.norm(x)
            cls_tok = x[:, 0]
            patch_toks = x[:, 1:]
            return cls_tok, patch_toks

    if args.grad_checkpoint and args.model == "dino":
        vit: Any = _CheckpointedPatchViT(
            img_size=args.img_size,
            patch=args.vit_patch,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
        ).to(device)
    else:
        vit = m.PatchViT(
            img_size=args.img_size,
            patch=args.vit_patch,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
        ).to(device)

    class _VitCLS(nn.Module):
        def __init__(self, vit: nn.Module) -> None:
            super().__init__()
            self.vit = vit

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cls, _patches = self.vit.forward_features(x)
            return cls

    student = None
    teacher = None

    # Default AMP on for DINO mode unless explicitly disabled.
    if args.model == "dino" and torch.cuda.is_available() and ("--amp" not in sys.argv):
        args.amp = True

    amp_dtype = None
    if args.amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    if args.model == "dino":
        student = m.DinoStudentTeacher(vit, out_dim=args.dino_out_dim).to(device)
        # Teacher must match student architecture exactly (including checkpointed ViT).
        teacher = copy.deepcopy(student).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        model = student

        def _loss_for(x: torch.Tensor) -> torch.Tensor:
            assert student is not None and teacher is not None
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(args.amp)):
                t_logits, _t_patches = teacher(x)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(args.amp)):
                s_logits, _s_patches = student(x)
                return F.mse_loss(s_logits, t_logits)

    else:
        head = _TinyHead(args.vit_dim).to(device)
        model = nn.Sequential(_VitCLS(vit), head)

        def _loss_for(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(args.amp)):
                y = model(x)
                return (y.float() ** 2).mean()

    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    summary_path = out_dir / "summary.json"

    fieldnames = [
        "batch_size",
        "grad_accum_steps",
        "effective_batch",
        "num_workers",
        "pin_memory",
        "status",
        "error",
        "images_per_s",
        "data_ms",
        "h2d_ms",
        "compute_ms",
        "step_ms",
        "mem_max_allocated_bytes",
        "mem_max_reserved_bytes",
        "cuda_free_bytes",
        "cuda_total_bytes",
        "cpu_maxrss_kb",
        "loadavg_1",
        "loadavg_5",
        "loadavg_15",
        "bound",
    ]

    results: list[dict[str, Any]] = []

    with csv_path.open("w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()

        combos: list[Combo] = []
        for bs in cfg["batch_sizes"]:
            for nw in cfg["num_workers"]:
                for pm in cfg["pin_memory"]:
                    combos.append(Combo(batch_size=bs, num_workers=nw, pin_memory=pm))

        for i, c in enumerate(combos):
            print(
                f"[{i+1}/{len(combos)}] bs={c.batch_size} accum={int(args.grad_accum_steps)} "
                f"eff={int(c.batch_size) * int(args.grad_accum_steps)} nw={c.num_workers} pin={bool(c.pin_memory)}",
                flush=True,
            )
            row: dict[str, Any] = {
                "batch_size": c.batch_size,
                "grad_accum_steps": int(args.grad_accum_steps),
                "effective_batch": int(c.batch_size) * int(args.grad_accum_steps),
                "num_workers": c.num_workers,
                "pin_memory": bool(c.pin_memory),
                "status": "error",
                "error": "",
                "images_per_s": 0.0,
                "data_ms": 0.0,
                "h2d_ms": 0.0,
                "compute_ms": 0.0,
                "step_ms": 0.0,
                "mem_max_allocated_bytes": 0,
                "mem_max_reserved_bytes": 0,
                "cuda_free_bytes": 0,
                "cuda_total_bytes": 0,
                "cpu_maxrss_kb": 0,
                "loadavg_1": 0.0,
                "loadavg_5": 0.0,
                "loadavg_15": 0.0,
                "bound": "unknown",
            }

            it = None
            dl = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                mi0 = _cuda_mem_info() or {}
                row["cuda_free_bytes"] = int(mi0.get("free", 0))
                row["cuda_total_bytes"] = int(mi0.get("total", 0))

                # Deterministic DataLoader ordering.
                gen = torch.Generator()
                gen.manual_seed(args.train_seed)

                def _worker_init_fn(worker_id: int) -> None:
                    random.seed(args.train_seed + worker_id)
                    try:
                        import numpy as np  # type: ignore

                        np.random.seed(args.train_seed + worker_id)
                    except Exception:
                        pass
                    torch.manual_seed(args.train_seed + worker_id)

                # NOTE: DataLoader prefetch can dominate memory on unified-memory systems.
                # Default prefetch_factor=2 means each worker can hold 2 *batches* in-flight.
                # For large models, that can push you over the cliff, especially at higher worker counts.
                dl_kwargs: dict[str, Any] = {}
                if c.num_workers > 0:
                    dl_kwargs["prefetch_factor"] = 1

                dl = torch.utils.data.DataLoader(
                    ds,
                    batch_size=c.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=c.num_workers,
                    pin_memory=bool(c.pin_memory),
                    generator=gen,
                    worker_init_fn=_worker_init_fn,
                    # Keep workers non-persistent since we create a fresh DataLoader per combo.
                    # This also reduces the chance of lingering processes at interpreter shutdown.
                    persistent_workers=False,
                    **dl_kwargs,
                )
                it = iter(dl)

                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

                grad_accum = max(1, int(args.grad_accum_steps))
                use_scaler = bool(args.amp and amp_dtype == torch.float16 and torch.cuda.is_available())
                scaler = (
                    torch.amp.GradScaler(device="cuda", enabled=use_scaler)
                    if torch.cuda.is_available()
                    else None
                )

                def _sync() -> None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                def _backward(loss: torch.Tensor) -> None:
                    loss = loss / float(grad_accum)
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                def _step() -> None:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()

                # Warmup (counts optimizer updates, not microbatches).
                for _ in range(max(0, int(args.warmup_steps))):
                    opt.zero_grad(set_to_none=True)
                    for micro in range(grad_accum):
                        batch = next(it)
                        x = batch.to(device, non_blocking=True)
                        loss = _loss_for(x)
                        _backward(loss)
                        if micro == grad_accum - 1:
                            _step()
                    _sync()

                data_s = 0.0
                h2d_s = 0.0
                compute_s = 0.0
                step_s = 0.0

                # Benchmark (counts optimizer updates, not microbatches).
                # We synchronize once per optimizer update to avoid massive sync overhead on ROCm.
                for _ in range(max(1, int(args.bench_steps))):
                    t0_step = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    for micro in range(grad_accum):
                        t0 = time.perf_counter()
                        batch = next(it)
                        t1 = time.perf_counter()

                        x = batch.to(device, non_blocking=True)
                        t2 = time.perf_counter()

                        loss = _loss_for(x)
                        _backward(loss)

                        if micro == grad_accum - 1:
                            _step()
                        t3 = time.perf_counter()

                        data_s += t1 - t0
                        h2d_s += t2 - t1
                        compute_s += t3 - t2

                    _sync()
                    t1_step = time.perf_counter()
                    step_s += t1_step - t0_step

                images = float(c.batch_size) * float(grad_accum) * float(max(1, int(args.bench_steps)))
                ips = images / max(step_s, 1e-9)

                mem = _torch_mem_snapshot() or {}
                row["mem_max_allocated_bytes"] = int(mem.get("max_allocated", 0))
                row["mem_max_reserved_bytes"] = int(mem.get("max_reserved", 0))

                try:
                    row["cpu_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                except Exception:
                    pass
                try:
                    la1, la5, la15 = os.getloadavg()
                    row["loadavg_1"] = float(la1)
                    row["loadavg_5"] = float(la5)
                    row["loadavg_15"] = float(la15)
                except Exception:
                    pass

                row["images_per_s"] = float(ips)
                row["data_ms"] = float(1000.0 * data_s)
                row["h2d_ms"] = float(1000.0 * h2d_s)
                row["compute_ms"] = float(1000.0 * compute_s)
                row["step_ms"] = float(1000.0 * step_s)

                # Very simple bottleneck classification.
                if data_s > compute_s:
                    row["bound"] = "io_bound"
                elif compute_s > data_s:
                    row["bound"] = "compute_bound"
                else:
                    row["bound"] = "mixed"

                row["status"] = "ok"

            except BaseException as e:
                if _is_oom(e):
                    row["status"] = "oom"
                else:
                    row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"

                # Capture failure-time memory state (useful when OOM is asynchronous).
                try:
                    mem = _torch_mem_snapshot() or {}
                    row["mem_max_allocated_bytes"] = int(mem.get("max_allocated", 0))
                    row["mem_max_reserved_bytes"] = int(mem.get("max_reserved", 0))
                except Exception:
                    pass
                try:
                    mi1 = _cuda_mem_info() or {}
                    row["cuda_free_bytes"] = int(mi1.get("free", row.get("cuda_free_bytes", 0) or 0))
                    row["cuda_total_bytes"] = int(mi1.get("total", row.get("cuda_total_bytes", 0) or 0))
                except Exception:
                    pass

                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            finally:
                # Ensure worker processes are shut down before moving to the next combo.
                try:
                    if it is not None and hasattr(it, "_shutdown_workers"):
                        it._shutdown_workers()  # type: ignore[attr-defined]
                except Exception:
                    pass
                it = None
                dl = None

                # Help the allocator reuse memory between combos (large models can fragment).
                try:
                    del opt
                except Exception:
                    pass
                try:
                    del scaler
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            results.append(row)
            w.writerow(row)
            fcsv.flush()

            # Persist incremental JSON after each combo.
            _write_json(json_path, {"env": env, "config": cfg, "results": results})

    # Summary: max stable microbatch/effective batch and best throughput.
    ok = [r for r in results if r.get("status") == "ok"]
    max_batch = max((int(r["batch_size"]) for r in ok), default=0)
    max_effective = max((int(r.get("effective_batch", 0)) for r in ok), default=0)
    best = None
    if ok:
        best = max(ok, key=lambda r: float(r.get("images_per_s", 0.0)))

    summary = {
        "max_stable_batch": int(max_batch),
        "max_stable_effective_batch": int(max_effective),
        "best": best,
    }
    _write_json(summary_path, summary)

    print(f"out_dir={out_dir}")
    print(f"max_stable_batch={summary['max_stable_batch']}")
    print(f"max_stable_effective_batch={summary['max_stable_effective_batch']}")
    if best is not None:
        print(
            "best="
            + json.dumps(
                {
                    "batch_size": best["batch_size"],
                    "grad_accum_steps": best.get("grad_accum_steps"),
                    "effective_batch": best.get("effective_batch"),
                    "num_workers": best["num_workers"],
                    "pin_memory": best["pin_memory"],
                    "images_per_s": best["images_per_s"],
                    "bound": best["bound"],
                }
            )
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

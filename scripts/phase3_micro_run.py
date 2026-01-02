#!/usr/bin/env python3
"""Phase 3: Micro-run (fail fast) training loop.

This script is intentionally *minimal* and self-contained:
- Loads preprocessed PNGs from Phase 2 (`data/processed/_index/index.csv`).
- Builds a small ViT + DINO-style student/teacher loop (EMA teacher).
- Optionally adds a Gram Anchoring regularizer computed from patch token features.
- Saves/resumes checkpoints and supports safe Ctrl+C interrupt handling.

Note: This is a micro-run harness, not the final Phase 5 trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _need(mod: str) -> None:
    raise SystemExit(f"Missing dependency: {mod}. Install it (e.g., into .venv) and retry.")


try:
    import numpy as np
except Exception:  # pragma: no cover
    _need("numpy")

try:
    from PIL import Image
except Exception:  # pragma: no cover
    _need("pillow")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    _need("torch")


@dataclass(frozen=True)
class RunConfig:
    img_size: int
    rw_level_min: float
    rw_level_max: float
    rw_width_min: float
    rw_width_max: float
    patch: int
    dim: int
    depth: int
    heads: int
    mlp_ratio: float
    out_dim: int
    batch_size: int
    lr: float
    weight_decay: float
    ema: float
    teacher_temp: float
    student_temp: float
    gram_enabled: bool
    gram_weight: float
    subset_seed: int
    subset_size: int
    train_seed: int
    sdp_backend: str
    index_csv: str


class PatchViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, H, W) with H=W=img_size
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, P, D)

        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos

        x = self.enc(x)
        x = self.norm(x)
        cls_tok = x[:, 0]
        patch_toks = x[:, 1:]
        return cls_tok, patch_toks


class DinoStudentTeacher(nn.Module):
    def __init__(self, vit: PatchViT, out_dim: int = 8192) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Sequential(
            nn.Linear(vit.dim, vit.dim),
            nn.GELU(),
            nn.Linear(vit.dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, patches = self.vit.forward_features(x)
        logits = self.head(cls)
        return logits, patches


def _ema_update(teacher: nn.Module, student: nn.Module, m: float) -> None:
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)


def _gram_matrix(patches: torch.Tensor) -> torch.Tensor:
    # patches: (B, P, D)
    x = F.normalize(patches, dim=-1)
    # covariance-ish: (B, D, D)
    x_t = x.transpose(1, 2)  # (B, D, P)
    g = (x_t @ x) / max(1, x.shape[1])
    return g


def _dino_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, t_s: float, t_t: float) -> torch.Tensor:
    # teacher is stop-grad
    p_t = F.softmax(teacher_logits / t_t, dim=-1)
    log_p_s = F.log_softmax(student_logits / t_s, dim=-1)
    return -(p_t * log_p_s).sum(dim=-1).mean()


@dataclass(frozen=True)
class IndexRow:
    png_path: Path
    series_dir: str
    slice_index: int
    encoding: str


def _load_index_rows(index_csv: Path) -> list[IndexRow]:
    if not index_csv.exists():
        raise SystemExit(f"index_csv not found: {index_csv}")

    out: list[IndexRow] = []
    with index_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if "png_path" not in fields:
            raise SystemExit(f"index_csv missing png_path column: {index_csv}")
        for row in r:
            out.append(
                IndexRow(
                    png_path=Path(row["png_path"]),
                    series_dir=row.get("series_dir", ""),
                    slice_index=int(row.get("slice_index", 0) or 0),
                    encoding=row.get("encoding", ""),
                )
            )
    if not out:
        raise SystemExit(f"No rows in index_csv: {index_csv}")
    return out


HU_OFFSET = 32768.0


def _window_hu_to_01(img_hu: "np.ndarray", level: float, width: float) -> "np.ndarray":
    lo = level - width / 2.0
    hi = level + width / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)


class PngDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[IndexRow],
        img_size: int,
        rw_level_min: float,
        rw_level_max: float,
        rw_width_min: float,
        rw_width_max: float,
    ) -> None:
        self.rows = rows
        self.img_size = img_size
        self.rw_level_min = rw_level_min
        self.rw_level_max = rw_level_max
        self.rw_width_min = rw_width_min
        self.rw_width_max = rw_width_max

        self._series_map: dict[str, dict[int, Path]] = {}
        self._series_minmax: dict[str, tuple[int, int]] = {}
        for r in rows:
            sm = self._series_map.setdefault(r.series_dir, {})
            sm[r.slice_index] = r.png_path
        for s, mp in self._series_map.items():
            if mp:
                ks = sorted(mp.keys())
                self._series_minmax[s] = (ks[0], ks[-1])

    def __len__(self) -> int:  # pragma: no cover
        return len(self.rows)

    def _load_hu01(self, p: Path, level: float, width: float) -> "np.ndarray":
        im = Image.open(p)
        if im.size != (self.img_size, self.img_size):
            im = im.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(im)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        hu = arr.astype(np.float32) - HU_OFFSET
        return _window_hu_to_01(hu, level=level, width=width)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.rows[idx]
        p = row.png_path

        # Legacy mode: baked RGB windows.
        try:
            mode = Image.open(p).mode
        except Exception:
            mode = ""

        is_hu16 = row.encoding.startswith("hu16") or mode.startswith("I")
        if not is_hu16:
            im = Image.open(p).convert("RGB")
            if im.size != (self.img_size, self.img_size):
                im = im.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        # HU16 mode: load z-1/z/z+1 as 3 channels, then apply random windowing.
        level = random.uniform(self.rw_level_min, self.rw_level_max)
        width = random.uniform(self.rw_width_min, self.rw_width_max)

        s = row.series_dir
        z = row.slice_index
        z0, z1 = self._series_minmax.get(s, (z, z))

        def _clamp(k: int) -> int:
            return max(z0, min(z1, k))

        mp = self._series_map.get(s, {})
        p_m1 = mp.get(_clamp(z - 1), p)
        p_0 = mp.get(_clamp(z), p)
        p_p1 = mp.get(_clamp(z + 1), p)

        a = self._load_hu01(p_m1, level=level, width=width)
        b = self._load_hu01(p_0, level=level, width=width)
        c = self._load_hu01(p_p1, level=level, width=width)

        x = np.stack([a, b, c], axis=0)
        return torch.from_numpy(x).contiguous()


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        st["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    if "python" in st:
        random.setstate(st["python"])
    if "torch" in st:
        torch.set_rng_state(st["torch"])
    if "numpy" in st:
        np.random.set_state(st["numpy"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_sdp_backend(kind: str) -> None:
    # NOTE: These APIs are under torch.backends.cuda even on ROCm builds.
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
        # Prefer mem_efficient, but keep math enabled as a safe fallback.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    raise SystemExit(f"unknown --sdp-backend: {kind}")


class _StopFlag:
    def __init__(self) -> None:
        self.stop = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", type=Path, default=Path("data/processed/_index/index.csv"))
    ap.add_argument("--run-dir", type=Path, default=Path("data/runs/phase3_micro_run"))
    ap.add_argument("--resume", type=Path, default=None, help="Checkpoint path; omit for no resume")

    ap.add_argument("--img-size", type=int, default=224)

    # Random windowing (only applied when Phase 2 data is HU16 grayscale slices).
    ap.add_argument("--rw-level-min", type=float, default=-700.0)
    ap.add_argument("--rw-level-max", type=float, default=100.0)
    ap.add_argument("--rw-width-min", type=float, default=300.0)
    ap.add_argument("--rw-width-max", type=float, default=2000.0)

    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--out-dim", type=int, default=8192)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)

    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--subset-size", type=int, default=1000)
    ap.add_argument("--train-seed", type=int, default=0, help="Seed for model init + DataLoader shuffle")
    ap.add_argument(
        "--sdp-backend",
        choices=["auto", "math", "mem_efficient"],
        default="auto",
        help="Force scaled_dot_product_attention backend (use 'math' for maximum correctness)",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=200)

    ap.add_argument("--gram", action="store_true", help="Enable Gram Anchoring")
    ap.add_argument("--gram-weight", type=float, default=1.0)

    ap.add_argument("--amp", action="store_true", help="Use autocast + GradScaler")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    _seed_all(args.train_seed)
    _set_sdp_backend(args.sdp_backend)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device.type}")
    print(f"torch_version={torch.__version__}")
    print(f"sdp_backend={args.sdp_backend}")
    if hasattr(torch.backends, "cuda"):
        try:
            print(
                "sdp_enabled="
                f"flash={torch.backends.cuda.flash_sdp_enabled()} "
                f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
                f"math={torch.backends.cuda.math_sdp_enabled()}"
            )
        except Exception:
            pass

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_index_rows(args.index_csv)
    rng = random.Random(args.subset_seed)
    rng.shuffle(all_rows)
    subset = all_rows[: min(args.subset_size, len(all_rows))]

    (run_dir / "subset.json").write_text(
        json.dumps(
            [
                {
                    "png_path": str(r.png_path),
                    "series_dir": r.series_dir,
                    "slice_index": r.slice_index,
                    "encoding": r.encoding,
                }
                for r in subset
            ],
            indent=2,
        )
        + "\n"
    )

    cfg = RunConfig(
        img_size=args.img_size,
        rw_level_min=args.rw_level_min,
        rw_level_max=args.rw_level_max,
        rw_width_min=args.rw_width_min,
        rw_width_max=args.rw_width_max,
        patch=args.patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        out_dim=args.out_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema=args.ema,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
        gram_enabled=bool(args.gram),
        gram_weight=args.gram_weight,
        subset_seed=args.subset_seed,
        subset_size=args.subset_size,
        train_seed=args.train_seed,
        sdp_backend=args.sdp_backend,
        index_csv=str(args.index_csv),
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    def _worker_init_fn(worker_id: int) -> None:
        _seed_all(args.train_seed + worker_id)

    gen = torch.Generator()
    gen.manual_seed(args.train_seed)

    ds = PngDataset(
        subset,
        img_size=args.img_size,
        rw_level_min=args.rw_level_min,
        rw_level_max=args.rw_level_max,
        rw_width_min=args.rw_width_min,
        rw_width_max=args.rw_width_max,
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=gen,
        worker_init_fn=_worker_init_fn,
    )
    it = iter(dl)

    vit_s = PatchViT(
        img_size=args.img_size,
        patch=args.patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    student = DinoStudentTeacher(vit_s, out_dim=args.out_dim).to(device)

    vit_t = PatchViT(
        img_size=args.img_size,
        patch=args.patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    teacher = DinoStudentTeacher(vit_t, out_dim=args.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and torch.cuda.is_available()))

    start_step = 0

    if args.resume is not None:
        ckpt_path = args.resume
        if not ckpt_path.is_file():
            raise SystemExit(f"resume checkpoint not found (or not a file): {ckpt_path}")
        # NOTE: PyTorch 2.6+ defaults `weights_only=True`, which rejects non-tensor
        # objects (we store RNG state + config). This checkpoint is local/trusted.
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(payload["student"], strict=True)
        teacher.load_state_dict(payload["teacher"], strict=True)
        opt.load_state_dict(payload["opt"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if payload.get("rng") is not None:
            _set_rng_state(payload["rng"])
        start_step = int(payload.get("step", 0))
        print(f"resume=true step={start_step} ckpt={ckpt_path}")

    stop = _StopFlag()

    def _handle_sigint(_sig: int, _frame: Any) -> None:
        stop.stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    t0 = time.time()
    last = t0

    for step in range(start_step, args.steps):
        if stop.stop:
            print("interrupt=true")
            break

        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        x = batch.to(device, non_blocking=True)

        with torch.no_grad():
            t_logits, t_patches = teacher(x)

        with torch.cuda.amp.autocast(enabled=bool(args.amp and torch.cuda.is_available())):
            s_logits, s_patches = student(x)
            loss_dino = _dino_loss(s_logits, t_logits, t_s=args.student_temp, t_t=args.teacher_temp)

            loss_gram = torch.tensor(0.0, device=device)
            if args.gram:
                g_s = _gram_matrix(s_patches)
                g_t = _gram_matrix(t_patches)
                loss_gram = F.mse_loss(g_s, g_t)

            loss = loss_dino + (args.gram_weight * loss_gram if args.gram else 0.0)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        _ema_update(teacher, student, m=args.ema)

        if (step + 1) % 10 == 0:
            now = time.time()
            dt = now - last
            last = now
            imgs_s = (args.batch_size * 10) / max(1e-6, dt)
            print(
                f"step={step+1} loss={loss.item():.6f} dino={loss_dino.item():.6f} "
                f"gram={loss_gram.item():.6f} imgs_per_s={imgs_s:.2f} device={device.type}"
            )

        if (step + 1) % args.ckpt_every == 0:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_path = ckpt_dir / f"step_{step+1:08d}.pth"
            latest = ckpt_dir / "latest.pth"
            payload = {
                "step": step + 1,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "rng": _rng_state(),
                "config": asdict(cfg),
            }
            _save_checkpoint(ckpt_path, payload)
            _save_checkpoint(latest, payload)
            print(f"checkpoint=true step={step+1} path={ckpt_path}")

    # Always write a final checkpoint on exit/interrupt.
    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "latest.pth"
    payload = {
        "step": step + 1 if "step" in locals() else start_step,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "rng": _rng_state(),
        "config": asdict(cfg),
    }
    _save_checkpoint(latest, payload)

    elapsed = time.time() - t0
    print("ok=true")
    print(f"run_dir={run_dir}")
    print(f"elapsed_s={elapsed:.1f}")
    print(f"latest_ckpt={latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

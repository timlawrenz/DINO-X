#!/usr/bin/env python3
"""Baseline: CIFAR-10 DINO-style pretraining.

Goal: establish an objective, non-medical baseline that proves the training loop
(student/teacher, centering, 2-view cross-view prediction) works end-to-end.

Artifacts are written under data/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Reuse Phase 5 reference implementations (DINO loss w/ centering, ViT backbone).
from scripts.phase5_big_run import DINOLoss, DinoStudentTeacher, PatchViT, get_lr


@dataclass(frozen=True)
class CifarVitConfig:
    img_size: int = 32
    patch: int = 4
    dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    out_dim: int = 8192


@dataclass(frozen=True)
class RunConfig:
    model: CifarVitConfig
    seed: int
    device: str
    batch_size: int
    accumulation_steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    max_steps: int
    ema: float
    teacher_temp: float
    student_temp: float
    amp: bool
    num_workers: int
    ckpt_every: int
    gram_enabled: bool
    gram_weight: float
    data_root: str
    created_at: str

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        st["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return st


def _set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu() if isinstance(st["torch"], torch.Tensor) else st["torch"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu() if isinstance(s, torch.Tensor) else s for s in st["cuda"]])


class TwoCrops:
    def __init__(self, t1: transforms.Compose, t2: transforms.Compose) -> None:
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img):
        return [self.t1(img), self.t2(img)]


def _cifar_views(img_size: int) -> TwoCrops:
    # CIFAR-10 defaults (RGB) + standard self-supervised-ish augmentations.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    base = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    t1 = transforms.Compose(base)
    t2 = transforms.Compose(base)
    return TwoCrops(t1, t2)


def _compute_gram_anchoring_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    # feats: (B, N+1, D) from PatchViT; skip CLS.
    s = F.normalize(student_feats[:, 1:], p=2, dim=-1)
    t = F.normalize(teacher_feats[:, 1:], p=2, dim=-1)
    s_gram = torch.bmm(s, s.transpose(1, 2))
    t_gram = torch.bmm(t, t.transpose(1, 2))
    return F.mse_loss(s_gram, t_gram)


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dino_loss: DINOLoss,
    cfg: RunConfig,
) -> None:
    payload = {
        "step": int(step),
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "dino_loss": dino_loss.state_dict(),
        "rng": _get_rng_state(),
        "config": asdict(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dino_loss: DINOLoss,
    device: torch.device,
) -> tuple[int, RunConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(payload["student"], strict=True)
    teacher.load_state_dict(payload["teacher"], strict=True)
    opt.load_state_dict(payload["opt"])
    if payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if payload.get("dino_loss") is not None:
        dino_loss.load_state_dict(payload["dino_loss"])
    if payload.get("rng") is not None:
        _set_rng_state(payload["rng"])

    cfg_dict = payload.get("config") or {}
    model_dict = cfg_dict.get("model") or {}
    model = CifarVitConfig(**model_dict)
    cfg = RunConfig(model=model, **{k: v for k, v in cfg_dict.items() if k != "model"})
    return int(payload.get("step", 0)), cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline CIFAR-10 DINO pretraining")

    ap.add_argument("--run-root", type=Path, default=Path("data/runs/baseline_cifar10"))
    ap.add_argument("--data-root", type=Path, default=Path("data/_torchvision"))
    ap.add_argument("--resume", type=Path, default=None)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--out-dim", type=int, default=8192)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accumulation-steps", type=int, default=1)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--max-steps", type=int, default=20000)

    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--ckpt-every", type=int, default=1000)

    ap.add_argument("--gram", action="store_true", help="Enable Gram Anchoring (A/B validation run)")
    ap.add_argument("--gram-weight", type=float, default=1.0)

    args = ap.parse_args()

    _seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = CifarVitConfig(
        img_size=int(args.img_size),
        patch=int(args.patch),
        dim=int(args.dim),
        depth=int(args.depth),
        heads=int(args.heads),
        mlp_ratio=float(args.mlp_ratio),
        out_dim=int(args.out_dim),
    )

    cfg = RunConfig(
        model=model_cfg,
        seed=int(args.seed),
        device=str(device),
        batch_size=int(args.batch_size),
        accumulation_steps=int(args.accumulation_steps),
        lr=float(args.lr),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        ema=float(args.ema),
        teacher_temp=float(args.teacher_temp),
        student_temp=float(args.student_temp),
        amp=bool(args.amp),
        num_workers=int(args.num_workers),
        ckpt_every=int(args.ckpt_every),
        gram_enabled=bool(args.gram),
        gram_weight=float(args.gram_weight),
        data_root=str(args.data_root),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    # Data
    views = _cifar_views(model_cfg.img_size)
    ds = datasets.CIFAR10(root=str(args.data_root), train=True, download=True, transform=views)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    it = iter(dl)

    # Model
    vit_s = PatchViT(
        img_size=model_cfg.img_size,
        patch=model_cfg.patch,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_ratio=model_cfg.mlp_ratio,
        use_grad_checkpoint=False,
    )
    student = DinoStudentTeacher(vit_s, out_dim=model_cfg.out_dim).to(device)

    vit_t = PatchViT(
        img_size=model_cfg.img_size,
        patch=model_cfg.patch,
        dim=model_cfg.dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
        mlp_ratio=model_cfg.mlp_ratio,
        use_grad_checkpoint=False,
    )
    teacher = DinoStudentTeacher(vit_t, out_dim=model_cfg.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    dino_loss_fn = DINOLoss(model_cfg.out_dim).to(device)

    start_step = 0

    if args.resume is not None:
        start_step, loaded_cfg = _load_checkpoint(
            args.resume,
            student=student,
            teacher=teacher,
            opt=opt,
            scaler=scaler,
            dino_loss=dino_loss_fn,
            device=device,
        )
        # Keep logging under the new run_dir even if resuming.
        print(f"resume=true step={start_step} checkpoint={args.resume}")
        # Persist loaded config for provenance.
        (run_dir / "resumed_from.json").write_text(json.dumps(asdict(loaded_cfg), indent=2) + "\n")

    tb = SummaryWriter(log_dir=str(run_dir))
    print(f"device={device.type}")
    print(f"run_dir={run_dir}")
    print(f"effective_batch_size={cfg.effective_batch_size}")

    loss_hist: list[float] = []
    t0 = time.time()
    last_log = t0

    for step in range(start_step, cfg.max_steps):
        # LR schedule
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        try:
            views, _y = next(it)
        except StopIteration:
            it = iter(dl)
            views, _y = next(it)

        # views is [batch_v1, batch_v2]
        batch = torch.cat(views, dim=0).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            s_feats = student.backbone(batch)
            with torch.no_grad():
                t_feats = teacher.backbone(batch)

            s_out = student.head(s_feats[:, 0])
            t_out = teacher.head(t_feats[:, 0])

            loss_dino = dino_loss_fn(s_out, t_out, cfg.student_temp, cfg.teacher_temp)
            loss = loss_dino

            loss_gram = torch.tensor(0.0, device=device)
            if cfg.gram_enabled:
                loss_gram = _compute_gram_anchoring_loss(s_feats, t_feats)
                loss = loss + cfg.gram_weight * loss_gram

            loss = loss / cfg.accumulation_steps

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step={step}: {loss.item()}")

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(cfg.ema).add_(p_s.data, alpha=1.0 - cfg.ema)

        loss_val = float(loss.item() * cfg.accumulation_steps)
        loss_hist.append(loss_val)

        if time.time() - last_log >= 10.0 or step == start_step:
            elapsed = time.time() - t0
            steps_s = (step - start_step + 1) / max(elapsed, 1e-6)
            samples_s = steps_s * cfg.effective_batch_size

            with torch.no_grad():
                # Under AMP, softmax output may be float16 and `clamp_min(1e-12)` becomes 0,
                # yielding 0 * log(0) = NaN. Do entropy math in fp32.
                t_prob = F.softmax(((t_out - dino_loss_fn.center).float()) / cfg.teacher_temp, dim=-1)
                s_prob = F.softmax((s_out.detach().float()) / cfg.student_temp, dim=-1)
                t_ent = (-t_prob * torch.log(t_prob.clamp_min(1e-12))).sum(dim=-1).mean().item()
                s_ent = (-s_prob * torch.log(s_prob.clamp_min(1e-12))).sum(dim=-1).mean().item()

                # Simple collapse proxy: std of CLS embeddings across the batch.
                cls = s_feats[:, 0].detach()
                emb_std = cls.std(dim=0).mean().item()

            print(
                f"step={step:6d} loss={loss_val:.4f} lr={lr:.2e} "
                f"steps/s={steps_s:.2f} samples/s={samples_s:.1f} "
                f"ent_t={t_ent:.3f} ent_s={s_ent:.3f} emb_std={emb_std:.4f}"
            )

            tb.add_scalar("Train/Loss_Total", loss_val, step)
            tb.add_scalar("Train/Loss_DINO", float(loss_dino.item()), step)
            tb.add_scalar("Train/Loss_Gram", float(loss_gram.item()), step)
            tb.add_scalar("Train/Entropy_Teacher", float(t_ent), step)
            tb.add_scalar("Train/Entropy_Student", float(s_ent), step)
            tb.add_scalar("Train/EmbeddingStd", float(emb_std), step)
            tb.add_scalar("Train/LR", float(lr), step)
            tb.add_scalar("Perf/Samples_Per_Sec", float(samples_s), step)
            tb.flush()
            last_log = time.time()

        if cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
            ckpt = run_dir / f"checkpoint_{step+1:08d}.pth"
            _save_checkpoint(
                ckpt,
                step=step + 1,
                student=student,
                teacher=teacher,
                opt=opt,
                scaler=scaler,
                dino_loss=dino_loss_fn,
                cfg=cfg,
            )
            print(f"checkpoint_saved={ckpt}")

    tb.close()

    final = run_dir / f"checkpoint_final_{cfg.max_steps:08d}.pth"
    _save_checkpoint(
        final,
        step=cfg.max_steps,
        student=student,
        teacher=teacher,
        opt=opt,
        scaler=scaler,
        dino_loss=dino_loss_fn,
        cfg=cfg,
    )

    metrics = {
        "kind": "cifar10_pretrain",
        "version": 1,
        "run_dir": str(run_dir),
        "final_checkpoint": str(final),
        "max_steps": int(cfg.max_steps),
        "loss_last": float(loss_hist[-1]) if loss_hist else None,
        "created_at": cfg.created_at,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("ok=true")
    print(f"run_dir={run_dir}")
    print(f"final_checkpoint={final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

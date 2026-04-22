
import math
import sys
from pathlib import Path

# Ensure repository root is importable when running as `python scripts/check_checkpoint.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def _tail_tfevents(run_dir: Path) -> None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        raise SystemExit(f"Failed to import tensorboard EventAccumulator: {e}")

    evs = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not evs:
        print(f"No TensorBoard event files found in {run_dir}")
        return

    ev = evs[-1]
    print(f"Loading TensorBoard events: {ev}")

    acc = EventAccumulator(str(ev))
    acc.Reload()

    # Common tags we care about for the entropy-wall diagnosis.
    tags = [
        "Train/Loss_Total",
        "Train/Entropy_Teacher",
        "Train/Entropy_Student",
        "Train/EmbeddingStd",
        "Train/LR",
    ]

    for tag in tags:
        if tag not in acc.Tags().get("scalars", []):
            continue
        s = acc.Scalars(tag)
        if not s:
            continue
        last = s[-1]
        print(f"{tag}: step={last.step} value={last.value}")


def check(ckpt_path: str) -> None:
    p = Path(ckpt_path)
    if p.is_dir():
        _tail_tfevents(p)
        return

    device = torch.device("cpu")
    print(f"Loading {p} on {device}")

    payload = torch.load(p, map_location=device, weights_only=False)

    cfg = payload.get("config") or {}
    model = (cfg.get("model") or {})
    out_dim = model.get("out_dim")
    if out_dim:
        out_dim = int(out_dim)
        print(f"out_dim={out_dim} ln(out_dim)={math.log(out_dim):.6f}")

    if "dino_loss" in payload:
        center = payload["dino_loss"]["center"]
        print(
            f"Center stats: min={center.min().item():.6f}, max={center.max().item():.6f}, "
            f"mean={center.mean().item():.6f}, std={center.std().item():.6f}"
        )
        if center.std() < 1e-6:
            print("WARNING: Center is uniform!")

    student_sd = payload["student"]
    weight_name = "head.4.weight" if "head.4.weight" in student_sd else list(student_sd.keys())[-1]
    weights = student_sd[weight_name]
    print(
        f"Weights {weight_name} stats: min={weights.min().item():.6f}, max={weights.max().item():.6f}, "
        f"mean={weights.mean().item():.6f}, std={weights.std().item():.6f}"
    )

    if "step" in payload:
        print(f"Checkpoint step: {payload['step']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(sys.argv[1])
    else:
        print("Usage: python scripts/check_checkpoint.py <checkpoint.pth | run_dir>")

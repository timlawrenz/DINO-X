
import sys
from pathlib import Path

# Ensure repository root is importable when running as `python scripts/check_checkpoint.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.phase5_big_run import ModelConfig, HardwareConfig, TrainingConfig, DinoStudentTeacher, PatchViT, DINOLoss

def check(ckpt_path):
    device = torch.device("cpu")
    print(f"Loading {ckpt_path} on {device}")
    
    # Load checkpoint
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check center
    if "dino_loss" in payload:
        center = payload["dino_loss"]["center"]
        print(f"Center stats: min={center.min().item():.6f}, max={center.max().item():.6f}, mean={center.mean().item():.6f}, std={center.std().item():.6f}")
        
        # Check if center is uniform
        # If all values in center are the same, std will be 0.
        if center.std() < 1e-6:
            print("WARNING: Center is uniform!")
    
    # Check student weights
    student_sd = payload["student"]
    # Check some weights
    weight_name = "head.4.weight" if "head.4.weight" in student_sd else list(student_sd.keys())[-1]
    weights = student_sd[weight_name]
    print(f"Weights {weight_name} stats: min={weights.min().item():.6f}, max={weights.max().item():.6f}, mean={weights.mean().item():.6f}, std={weights.std().item():.6f}")

    if "step" in payload:
        print(f"Checkpoint step: {payload['step']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(sys.argv[1])
    else:
        # Try to find latest
        run_dir = Path("data/runs/20260103_175300")
        ckpts = sorted(run_dir.glob("checkpoint_*.pth"))
        if ckpts:
            check(ckpts[-1])
        else:
            print("No checkpoints found")

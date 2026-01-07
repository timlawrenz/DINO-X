#!/usr/bin/env python3
"""
Summarize training runs from data/runs into a CSV report.
Preserves user notes added to the 'Notes' column in the CSV.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Path to runs directory (symlink aware)
RUNS_DIR = Path("data/runs")
OUTPUT_CSV = Path("docs/experiments.csv")
OUTPUT_MD = Path("docs/EXPERIMENTS.md")

def load_existing_notes(csv_path: Path) -> dict[str, str]:
    """Load existing notes from the CSV file."""
    notes = {}
    if not csv_path.exists():
        return notes
    
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Run ID" in row and "Notes" in row:
                    notes[row["Run ID"]] = row["Notes"]
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
    return notes

def parse_run(run_dir: Path) -> dict | None:
    """Parse a run directory and return a summary dict."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return None

    # Extract fields
    model = cfg.get("model", {})
    hw = cfg.get("hardware", {})
    
    # Calculate effective batch size
    bs = cfg.get("batch_size", 0)
    accum = cfg.get("accumulation_steps", 1)
    eff_bs = bs * accum

    # Determine status
    status = "Unknown"
    if (run_dir / "checkpoint_final.pth").exists() or any(run_dir.glob("checkpoint_final_*.pth")):
        status = "Completed"
    elif any(run_dir.glob("emergency_checkpoint*.pth")):
        status = "Crashed"
    else:
        # Check if modified recently (within 1 hour)
        try:
            mtime = config_path.stat().st_mtime
            import time
            if time.time() - mtime < 3600:
                status = "Running"
            else:
                status = "Stopped"
        except:
            pass

    return {
        "Run ID": run_dir.name,
        "Date": cfg.get("created_at", ""),
        "Model": model.get("name", "unknown"),
        "Patch": model.get("patch", ""),
        "Dim": model.get("dim", ""),
        "Batch": bs,
        "Accum": accum,
        "Eff Batch": eff_bs,
        "LR": cfg.get("lr", ""),
        "Min LR": cfg.get("min_lr", ""),
        "Warmup": cfg.get("warmup_steps", ""),
        "T-Temp": cfg.get("teacher_temp", ""),
        "Gram": cfg.get("gram_weight", ""),
        "Device": hw.get("device_name", "unknown"),
        "Status": status,
        "Commit": cfg.get("git_commit", "")[:7] if cfg.get("git_commit") else "",
    }

def main():
    if not RUNS_DIR.exists():
        print(f"Error: {RUNS_DIR} does not exist.")
        return

    # 1. Load existing notes
    existing_notes = load_existing_notes(OUTPUT_CSV)
    
    # 2. Scan runs
    runs = []
    # Use os.scandir to handle potential symlink issues gracefully if Path iterator fails
    # But Path.iterdir() should work if it points to a dir.
    # Note: os.listdir(RUNS_DIR) might be safer if RUNS_DIR is a broken link, but here it works.
    
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)
    
    print(f"Found {len(run_dirs)} run directories in {RUNS_DIR}...")
    
    for d in run_dirs:
        data = parse_run(d)
        if data:
            # Restore note
            data["Notes"] = existing_notes.get(data["Run ID"], "")
            runs.append(data)

    if not runs:
        print("No valid runs found (checked for config.json).")
        return

    # 3. Write CSV
    fieldnames = [
        "Run ID", "Date", "Model", "Patch", "Dim", "Batch", "Accum", "Eff Batch", 
        "LR", "Min LR", "Warmup", "T-Temp", "Gram", "Status", "Notes", "Device", "Commit"
    ]
    
    # Ensure directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)
    
    print(f"Updated {OUTPUT_CSV}")

    # 4. Write Markdown (for viewing)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Experiments Log\n\n")
        f.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Header
        cols = ["Run ID", "Model", "Eff Batch", "LR", "Warmup", "T-Temp", "Status", "Notes"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        
        for r in runs:
            row = [
                f"`{r['Run ID']}`",
                f"{r['Model']} (p{r['Patch']})",
                str(r['Eff Batch']),
                str(r['LR']),
                str(r['Warmup']),
                str(r['T-Temp']),
                r['Status'],
                r['Notes']
            ]
            f.write("| " + " | ".join(row) + " |\n")
            
    print(f"Updated {OUTPUT_MD}")

if __name__ == "__main__":
    main()

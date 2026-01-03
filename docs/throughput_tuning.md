# Throughput Tuning & Model Selection Report

**Date:** January 3, 2026
**Hardware:** AMD Strix Halo (128GB Unified Memory, ~96GB VRAM slice)
**Context:** Phase 4.5 of Project DINO-X

## 1. Executive Summary

The primary goal of this phase was to determine the maximum viable model size and training configuration that fits within the **15-day compute budget**. 

Initial experiments targeted a **ViT-Giant** architecture (1.5B parameters). However, results showed a peak throughput of **~5.4 images/s**, which would require ~60 days to complete the target training schedule. 

Consequently, the project has **pivoted to a ViT-Large** architecture (300M parameters). Throughput tuning for ViT-Large demonstrated a stable performance of **~23.5 images/s**, allowing for a complete training run within approximately 14 days.

## 2. ViT-Giant Evaluation (The "Fail")

We attempted to fit a standard ViT-Giant (`patch=14`, `dim=1536`, `depth=40`, `heads=24`) into the Strix Halo memory.

*   **Constraint:** Without Gradient Checkpointing, the model OOM'd at Batch Size 6.
*   **Optimization:** Enabling Gradient Checkpointing and `bf16` AMP allowed larger physical batches (up to 36), but the compute bound was severe.
*   **Result:** The best configuration achieved only ~5.4 img/s.

| Run ID | Batch Size (Phys) | Grad Accum | Effective Batch | Checkpointing | Throughput | Bound |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260102_213649` | 4 | 1 | 4 | No | 2.71 img/s | Compute |
| `20260102_212828` | 8 | 1 | 8 | No | **OOM** | Memory |
| `overnight_accum8...` | 24 | 8 | 192 | **Yes** | 5.26 img/s | Compute |
| `overnight_accum8...` | 32 | 8 | 256 | **Yes** | **5.40 img/s** | Compute |

> **Conclusion:** While the Strix Halo *can* fit the ViT-Giant in memory (using checkpointing), the compute throughput is insufficient for the project timeline.

## 3. ViT-Large Tuning (The "Pivot")

We stepped down to a ViT-Large (`patch=14`, `dim=1024`, `depth=24`, `heads=16`). This drastically reduced compute density and memory pressure.

*   **Stability:** Excellent. Memory usage is low (~7-10GB allocated), leaving massive headroom for the system and data loader.
*   **Sweet Spot:** Physical Batch Size 64 with Accumulation 4 (Effective 256) yields the highest efficiency.
*   **Throughput:** consistently **>23 img/s**.

### Key Results (`vit_large_sweep`)

| Batch Size (Phys) | Grad Accum | Effective Batch | Num Workers | Pin Mem | Throughput | Memory (Alloc) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 48 | 4 | 192 | 4 | True | **23.54 img/s** | ~6.3 GB |
| 64 | 4 | 256 | 4 | True | **23.51 img/s** | ~6.8 GB |
| 96 | 2 | 192 | 4 | True | 23.22 img/s | ~7.8 GB |
| 128 | 2 | 256 | 4 | True | 22.46 img/s | ~8.9 GB |
| 160 | 2 | 320 | 4 | True | 22.40 img/s | ~10.1 GB |

> **Selected Configuration:** `Batch Size 64` / `Accum 4` (Effective 256). 
> This matches the target effective batch size of 256 required for DINO stability while maintaining peak throughput.

## 4. Hardware Utilization Analysis

Analysis of the `vit_large_sweep` logs confirms:

1.  **Compute Bound:** All successful runs report `bound: compute_bound`. This indicates the Data Loader (workers) and Memory Bandwidth are keeping up with the GPU. We are correctly maximizing the compute units.
2.  **Memory Headroom:** With only ~10GB of VRAM utilized for the model and batch, the 128GB Unified Memory is largely available for caching the dataset, potentially speeding up epochs 2+ via OS-level page caching.
3.  **Data Loader:** `num_workers=4` with `pin_memory=True` appears sufficient to saturate the compute. Increasing workers further (tested in other sweeps) did not yield significant gains and only increased CPU load.

## 5. Final Decision

**Proceed with Phase 5 (The Big Run) using ViT-Large.**

*   **Model:** ViT-Large (`dim=1024`, `depth=24`)
*   **Batch Size:** 64 (Physical)
*   **Gradient Accumulation:** 4
*   **Effective Batch Size:** 256
*   **Precision:** BF16 Mixed Precision
*   **Gradient Checkpointing:** Enabled (Optional for Large, but keeps memory low for system stability)
*   **Expected Daily Progress:** ~2,000,000 images/day

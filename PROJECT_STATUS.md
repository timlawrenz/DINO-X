# Project Status — DINO-X

**Last updated:** 2026-07-15
**Phase / status:** Phase 5 STALLED — ViT-Large pan-organ training (4 memory-related aborts on Strix Halo)

## Current state

ViT-Small is definitively saturated: the 5-dataset pan-organ run collapsed at 86K steps (teacher entropy 0.001). The pivot to ViT-Large (923M params) was learning strongly — teacher entropy reached 6.60 (target 6.78) at step 6,369 — but crashed 4 times due to ROCm allocator fragmentation on the Strix Halo's unified memory. The model was not collapsing; the crash was a memory issue.

Best result so far: ViT-Small 5-dataset bs256 at 81K steps → LoRA AUROC 0.697 on LIDC malignancy, view retrieval 63× random.

## Immediate blockers / next action

**Resume ViT-Large from step 5,000** with memory mitigation:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduce allocator fragmentation)
- `--batch-size 4 --accumulation-steps 64` (smaller physical batch, less peak activation memory)
- If crashes persist, try `--batch-size 8 --accumulation-steps 32` with gradient checkpointing

See `docs/phase6_large_model_resume.md` for the full abort analysis and validation plan.
Resume instructions: `runs/20260511_032957_5dataset-phase5-large-bs256/README_RESUME.md`

## Headline result so far

| Metric | Best Value | Arm | Verdict |
|---|---|---|---|
| LoRA AUROC (LIDC malignancy) | 0.710 | 4-dataset ViT-Small, 5K steps | GO — PENDING adversarial pass |
| LoRA AUROC (5-dataset) | 0.697 | 5-dataset ViT-Small bs256, 81K steps | GO — PENDING adversarial pass |
| View retrieval ratio | 63× | 5-dataset ViT-Small, 81K steps | GO |
| Scale embedding loss | 0.134 | 2-organ scale-aware, 5K steps | GO — 67× lower than baseline |
| Spacing R² | 0.876 | Local 5K-step scale-aware | GO |
| Dataset discrimination AUC | 1.000 | 5K-step scale-aware | GO |
| ViT-Small 100K run | Collapse at 86K | 5-dataset bs256 | KILL — capacity saturated |
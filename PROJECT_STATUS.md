# Project Status — DINO-X

**Last updated:** 2026-07-24
**Phase / status:** Phase 9 ACTIVE — Single-Organ Specialist Expansion

## Current state

Phase 8 concluded that view retrieval and spacing metrics are invalid gates for single-organ models. Phase 9 applies the proven Phase 6 LIDC-specialist recipe to `msd-colon` and `msd-hepatic-vessel` to verify that the single-organ strategy generalizes and avoids pan-organ capacity dilution.

The `msd-colon` pretraining run is actively running on `max395` (job `dc0369eac513`). The `msd-hepatic-vessel` job (`c6fc88347c57`) is queued next on the GPU scheduler, with an autonomous Hermes cron job (`dino-x-hepatic-launcher`) polling to launch it automatically when the colon run completes.

## Immediate next action

Monitor the `msd-colon` run and wait for the `msd-hepatic-vessel` run to auto-launch. Evaluate both against LoRA baselines upon completion.

## Headline result

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| LoRA AUROC (LIDC malignancy) | **0.728** | ≥ 0.720 | **PASS** |
| View retrieval (layer 9) | 37× | ≥ 40× | FAIL |
| View retrieval (layer 12, default) | 31× | ≥ 40× | FAIL |
| Training stability | No collapse, 289 img/s | — | GO |
| Baseline (ViT-Small 4-dataset) | AUROC 0.710 | — | — |

## Next action

1. **Scale to other organs** — apply the proven LIDC-specialist recipe to cq500 (brain), pancreas-ct, etc.
2. **Many-to-many matching** — relax 1-to-1 in view retrieval eval (arXiv:2604.23670)
3. **Cross-Modality Expansion** — MRI (`dinox-mri-vit-small`) and X-ray (`dinox-xray-vit-small`) models.
4. **HF Hub Release** — publish `dinox-ct-vit-small-v1` model card + safetensors.
5. **Adversarial Pass Retrospective** — audit all GO verdicts against the 4-question checklist.

## Headline result

| Metric | Value | Verdict |
|---|---|---|
| ViT-Large best view retrieval | 34× (step 25K) | FAIL — gate was 100× |
| Scanner fingerprinting (dataset AUC) | 0.981 | Architectural failure confirmed |
| Capacity dilution (colon↔vessel cosine) | 0.962 | Same as ViT-Small |
| Scale embedding proven | R² 0.980, counterfactual 0.444 | GO — carries forward |
| Memory mitigation solved | 46 img/s, 67h to 50K, no OOM | GO — proven recipe for Strix Halo |
| ViT-Small best AUROC (4-dataset) | 0.710 LIDC malignancy | Best overall result |
| arXiv 2607.16317 validation | Deterministic entropy estimators collapse by construction | Confirms teacher entropy was wrong metric |

## Immediate next action

The pan-organ pretraining objective (DINO + KoLeo) is structurally wrong. Three paths:

1. **Single-organ specialists** — Train scale-aware ViT-Base on organ-specific datasets (LIDC, cq500). Proven recipe, guaranteed to work. Lowest risk.
2. **Objective redesign** — Replace DINO with MAE (masked autoencoder) or add adversarial domain adaptation (gradient reversal on dataset ID). Breaks the scanner-fingerprinting deadlock. Medium risk.
3. **MC-Dropout uncertainty** — Replace teacher entropy with sampling-based uncertainty as the primary health metric. Addresses the monitoring failure but doesn't fix the objective. Complements either path.

See `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md` for full analysis.

## Artifacts

- `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md` — mandatory KILL artifact
- `runs/20260719_042301_5dataset-phase5-large-bs256-v2/` — full 50K run with 10 checkpoints
- `results/panorgan_step35000.json` — pre-collapse eval
- `results/panorgan_step50000.json` — zombie-phase eval
- Best checkpoint: step 25K at `runs/20260719_042301_5dataset-phase5-large-bs256-v2/checkpoint_00025000.pth`

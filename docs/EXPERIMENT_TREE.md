# Experiment Tree

Living workstream map — status tags: `[ACTIVE]`, `[CONCLUDED]`, `[TBD]`. Evidence lives in the ledger (`docs/EXPERIMENTS_AND_RESULTS.md`). Governance: `docs/experiment-structure.md`.

---

## Active

- **[ACTIVE] Tier 1 Release Packaging** (`tier1-release-packaging`)
  - Hepatic-vessel specialist: internal AUROC 0.9456 + **external CRLM validation** (slice 0.9413, patient 0.739, 197 new patients) — release-ready.
  - LIDC specialist PASSES (AUROC 0.728) — ready for HF Hub.
  - Colon specialist FAILS (AUROC 0.6529) — needs new hypothesis before re-run.
  - Next: package hepatic-vessel + LIDC for HF Hub release.
  - **Release-readiness gates defined (5-gate framework).** Hepatic-vessel v1: Gates 2 & 4 PASS; **Gate 1 (stupid-simple inference) CLOSED** via `zoo/predict.py` (`load_classifier` + `predict_proba`, trained −30/120 window baked in; verified on max395: real CRLM slices separate present 0.994 vs absent 0.068). Gate 3 (public repro) deferred to v1.1 (CRLM licensing). **Gate 5 PASS** (citation added, commit `2f4cf18`).
  - **Public-flip blockers: NONE remaining in code.** ~~`zoo` not pip-installable~~ CLOSED (`pyproject.toml`, commit `865a090` — `pip install .` verified, import resolves with no PYTHONPATH). ~~`requirements.txt` pins local ROCm wheels~~ CLOSED (`requirements-inference.txt`, portable, no machine-specific paths). ~~citation empty~~ CLOSED (commit `2f4cf18`). Gate 5 now PASS. **Ready to push updated card to HF + flip public.** GitHub issue still pending user approval.
  - Git commit: `4e9571c` (zoo.predict one-liner), `2f4cf18` (citation), `9118f53` (external adversarial-pass fixes)
  - **External red-team (Gemini) passed after fixes.** 6 claims reviewed; 4 doc fixes applied, 1 verified non-issue (no uint16 overflow — max encodable HU +3276.7, not the 4000 clip bound), 1 real latent finding.
  - **hu16 encode/decode inconsistency RECONCILED (commit `a04c933`).** Canonical: encode `u16 = round(clip(HU,−1000,4000)*10) + 32768` (clamped to uint16), decode `HU = (u16−32768)*0.1`. Fixed the outliers — encode: `phase2_preprocess_{nifti,lidc_idri}.py` (was ×1 + overflow at +4000); decode: `phase3_micro_run.py`, `phase4_monitor.py`, `phase2_validate_samples.py` (was ×1); docs: `DATA_SOURCES.md`, `data_preprocessing.md`. 8 regression tests added (`tests/test_hu16_encoding.py`, all pass; full suite 185 pass, 9 pre-existing peft-env fails unrelated). Released model unaffected — train+eval were already self-consistent at ×0.1.

## Concluded

- **[CONCLUDED — GO with caveat] External Held-Out Validation (CRLM)** (`external-validation-crlm`)
  - Hepatic-vessel specialist evaluated on CRLM (TCIA colorectal liver metastases), 197 patients / 17,639 slices, genuinely external to MSD pretraining.
  - Slice-level AUROC 0.9413, patient-level AUROC 0.739 (majority vote), Spearman(vessel-richness, score) 0.53 (p≈1e-15).
  - Kills scanner-fingerprinting fear: model transfers to a different institution's scans on correct vessel semantics.
  - Patient-level "binary AUROC" 0.47 initially flagged — artifact of ill-posed all-mixed patient task, not a model failure.
  - Git commit: `2073ff0` + `a556689` (extractor + eval harness)

- **[CONCLUDED — PIVOT] Single-Organ Specialist Expansion** (`single-organ-specialists-expansion`)
  - Hepatic-vessel specialist: AUROC 0.9456 (PASS, best in project)
  - Colon specialist: AUROC 0.6529 (FAIL, unstable training)
  - Strategy validated but not universal — organ texture distinctiveness matters.
  - Git commit: `3880452` (DataLoader fix + label extractor)

- **[CONCLUDED — PIVOT] Scale LIDC Recipe to Other Organs** (`single-organ-specialists-expansion`)
  - The ViT-Base scale-aware recipe on LIDC-only produced the project's best malignancy AUROC (0.728).
  - Hypothesis: This single-organ specialist approach will generalize to other datasets (msd-colon, msd-hepatic-vessel), breaking the pan-organ capacity dilution.
  - Pre-registered gate: Since view retrieval and spacing metrics are invalid for single-organ models, these will be gated purely on their LoRA AUROC against a baseline (if available) or raw classification capability.
  - Data: `msd-colon` (38K slices) and `msd-hepatic-vessel` (48K slices) subsets extracted.
  - Git commit: `647efd9`

## Concluded

- **[CONCLUDED — PIVOT] Replace View Retrieval Gate for Single-Organ Models** (`replace-view-retrieval-gate`)
  - Tested spacing metrics as replacement gates on LIDC specialist (known good AUROC 0.728).
  - Failed targets (counterfactual 0.239 < 0.30, R² 0.941 < 0.95). Single datasets lack spacing variance to drive scale embeddings to pan-organ levels.
  - **Verdict:** Spacing gates invalid for single datasets. View retrieval invalid (architecture handicapped).
  - **New Governance:** Single-organ models gated **solely by LoRA AUROC** vs baseline.

- **[CONCLUDED — KILL] Positional Bias Projection for View Retrieval** (`positional-bias-projection`)
  - Noise-image PCA bias projection applied to layer 9 features on 50K checkpoint.
  - Result: **37×** — identical to layer 9 without bias projection. No improvement.
  - Hypothesis that DINO positional artifact degrades CLS-token view retrieval was disproven for LIDC CT.
  - Added `--pos-bias-project` flag to eval script.

- **[CONCLUDED — KILL] Phase 5: ViT-Large Pan-Organ Pretraining** (`runs/20260719_042301_5dataset-phase5-large-bs256-v2/`)
  - 5-dataset pan-organ corpus (400K slices), ViT-Large (923M params), effective batch 256
  - Resumed 2026-07-19, completed 2026-07-22 at step 50,000. Memory mitigation stable; run finished cleanly.
  - View retrieval peaked at 34× (step 25K) vs pre-registered 100× gate. Entropy collapsed to chaotic oscillation at 40K. Pan-organ eval confirmed scanner fingerprinting (AUC 0.981) and capacity dilution (colon↔vessel cosine 0.962).
  - **Root cause:** DINO + KoLeo objective rewards scanner identity and spacing geometry over cross-organ anatomy. ViT-Small and ViT-Large both hit same wall at different speeds — model size is not the bottleneck.
  - See `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`.

- **[CONCLUDED — GO] MVP Two-Organ Scale-Aware Ablation** (`runs/mvp-two-organ/`)
  - Baseline (no scale) vs Scale-Aware ViT-Small, 43K slices (LIDC + Pancreas-CT), 5K steps
  - Scale-Aware: 67× lower loss (0.134 vs 8.992), both arms healthy. View retrieval 5.0× vs 4.0×.

- **[CONCLUDED — GO] Pan-Organ Evaluation Protocol Validation** (`runs/eval_validation/`)
  - 6-metric eval suite validated on 1K-step checkpoints
  - Scale-aware: R²=0.724 spacing prediction, AUC=1.000 dataset discrimination

- **[CONCLUDED — GO] Local 5K-Step Scale-Aware** (`runs/v1-local-5k/`)
  - ViT-Small on RTX 2070 SUPER, 12 min training
  - View retrieval 14× (LIDC), spacing R²=0.876, cross-centroid cosine 0.164, 184× spacing sensitivity improvement

- **[CONCLUDED — GO] 4-Dataset Anti-Memorization Ablation** (`runs/vit-small-4dataset-ablation/`)
  - Fixed catastrophic collapse with KoLeo(0.1) + crop=0.3 + z-stride=3 + diverse batches
  - View retrieval 7.0× at 5K steps. Cleared architecture for pan-organ scaling.

- **[CONCLUDED — GO] LIDC Malignancy LoRA Benchmark** (`adapters/lidc-malignancy-lora-r8-64px-lung-window/`)
  - 4-dataset backbone, LoRA rank=8, 64px crops, lung HU window → AUROC 0.710
  - Crop size dominates (64px >> 128px >> 224px). Lung window adds +3% vs generic.

- **[CONCLUDED — GO] 5-Dataset Phase 3 Pretraining** (`runs/20260422_202622_5dataset-phase3-small/`)
  - 400K slices, ViT-Small, 50K steps, effective batch 128. View retrieval 56×.
  - Training stable with temperature-scaled sampling (T=2.0). bfloat16, no NaN events.

- **[CONCLUDED — GO] LoRA Benchmark: 5-Dataset vs 4-Dataset** (`adapters/lidc-malignancy-5dataset-*`)
  - 5-dataset AUROC 0.680 vs 4-dataset 0.710. Capacity dilution confirmed.
  - More pretraining steps degraded lung-specific features (10K > 25K > 50K).

- **[CONCLUDED — GO] 5-Dataset Doubled Batch Size (bs256)** (`runs/20260423_171906_5dataset-phase3-small-bs256/`)
  - Effective batch 128→256. Loss plateaued at ~35K. LoRA AUROC 0.684, view retrieval 54×.
  - Doubling batch did not close the 0.026 gap to 4-dataset specialist.

- **[CONCLUDED — GO] Ablation Scan: 5-Dataset LoRA Interventions** (`adapters/ablation-5dataset-bs256-*`)
  - Rank 16: 0.685 AUROC (best). Rank 32: overfits. Unfreeze 1 block: hurts (-0.009). 128px crops: -0.079.
  - No LoRA-side intervention bridges the gap. The gap is structural (backbone capacity).

- **[CONCLUDED — KILL] ViT-Small 100K Extended Pretraining** (`runs/20260428_*_5dataset-phase3-small-bs256-100k/`)
  - 5-dataset, cosine schedule restretched to 100K. 81K: AUROC 0.697 (climbing), view retrieval 63×.
  - 86K: catastrophic capacity collapse (teacher entropy 0.001, AUROC 0.642).
  - ViT-Small definitively saturated. Path forward: ViT-Base or ViT-Large.
  - See `docs/DISCONTINUATION_NOTICE_vit-small-100k.md` (to be created).

### Historical (Single-Dataset LIDC-IDRI, Jan 2026)

- **[CONCLUDED — GO] Ice Age / Low LR experiments** (`20260108_203723_4090_IceAge`, `20260109_104007_4090_LowLR_IceAge`)
  - ViT-Large, LIDC-only, effective batch 256. Found "Golden Zone" LR ~2e-5. Broken entropy wall (loss 6.78) with frozen teacher (0.9995) + sharp temp (0.02).
- Multiple additional ViT-Large single-dataset runs (Jan 3–9, 2026) — most Completed, some Stopped. See ledger for full table.

## TBD

- **[TBD] Fix `--ckpt-keep-last` Default** — change from 5 to 0 (keep all) or 10 to prevent checkpoint rotation from destroying intermediate evaluation points (lost 25K in Phase 6).
- **[TBD] Many-to-Many View Retrieval Matching** — relax 1-to-1 nearest neighbor to top-K mutual neighbors (arXiv:2604.23670) in `phase5_view_retrieval_eval.py`.
- **[TBD] Cross-Modality Expansion** — MRI (`dinox-mri-vit-small`) and X-ray (`dinox-xray-vit-small`) models.
- **[TBD] HF Hub Release** — publish `dinox-ct-vit-small-v1` model card + safetensors.
- **[TBD] Adversarial Pass Retrospective** — audit all GO verdicts against the 4-question checklist.
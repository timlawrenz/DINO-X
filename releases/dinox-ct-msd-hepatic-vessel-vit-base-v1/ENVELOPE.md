# dinox-ct-msd-hepatic-vessel-vit-base-v1 — Evidence Envelope

**Model:** Scale-aware ViT-Base (86M) single-organ specialist, self-supervised on
MSD Hepatic-Vessel CT, LoRA fine-tuned for hepatic/portal vessel detection.
**License:** Apache-2.0 · **HF Hub:** `timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1` (planned)
**Source branch:** `main` · **Built from commit:** `3880452` (backbone + adapter) and `d1aab1c` (external validation)

---

## Headline Results (Overview)

| Metric | Value | Where verified |
|---|---|---|
| Internal LoRA AUROC (MSD val) | **0.9456** | `adapters/msd-hepatic-vessel-vit-base-50k/finetune_config.json` |
| **External AUROC — TCIA CRLM** | **0.9413** (17,639 slices, 197 patients) | `reproduce/repro_metrics.py` + ledger |
| External patient-level AUROC (majority vote) | **0.739** | `reproduce/repro_metrics.py` + ledger |
| External Spearman(vessel-richness, score) | **0.527** (p≈1.8e-15) | `reproduce/repro_metrics.py` + ledger |

*Why the external number matters:* the internal 0.9456 was red-team flagged as
potentially inflated by pretraining leakage (the backbone had seen 93% of the
"held-out" fine-tuning patients). The salvage validated the model on TCIA CRLM —
a genuinely external, multi-institution dataset impossible to overlap with MSD.
Slice AUROC 0.9413 there confirms the model learned real vessel-tissue features,
not scanner identity. See `the_science/03_validation.md`.

---

## Follow The Science (reading order)

1. **`the_science/01_decision_trail.md`** — Why this model exists, and the pan-organ
   failure that made organ-specialists the right call. Includes the red-team leakage
   finding that forced the salvage.
2. **`the_science/02_training_recipe.md`** — Exact pretraining recipe (data, preprocessing,
   hyperparameters, scale-awareness) and LoRA fine-tuning config.
3. **`the_science/03_validation.md`** — Every test we ran and how we know it's real:
   the adversarial pass, the leakage audit, and the external CRLM validation.
4. **`the_science/04_negative_results.md`** — What failed on the way: the pan-organ
   capacity-dilution KILL, the invalid view-retrieval gates, the colon specialist FAIL.
5. **`the_science/05_known_limits.md`** — Where this model still fails, honestly.
6. **`reproduce/README.md`** — Reproduce every number yourself.

---

## Reproducibility Anchor

| Artifact | Git commit |
|---|---|
| Backbone checkpoint (train recipe) | `3880452` |
| LoRA adapter + head | on disk `adapters/msd-hepatic-vessel-vit-base-50k/` |
| CRLM label extractor | `a556689` |
| External eval harness | `21c280f` |
| Red-team report (motivated salvage) | `672454a` |
| External validation ledger entry | `d1aab1c` |

Every non-empty field here maps to an artifact that `reproduce/repro_metrics.py` can
recompute or that the ledger (`docs/EXPERIMENTS_AND_RESULTS.md`) records permanently.
Nothing is asserted from memory. See `provenance.yaml` for machine-readable details.
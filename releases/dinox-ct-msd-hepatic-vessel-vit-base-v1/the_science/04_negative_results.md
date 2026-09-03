# 04 — Negative Results: What We Tried That Did NOT Work

Published with the same rigor as successes. Each entry is falsifiable and points to its
evidence. These are the credibility assets — they tell you where the easy-looking path
was a trap.

---

## 1. Pan-organ foundation model — `[CONCLUDED — KILL]`

**Attempt:** One ViT-Small / ViT-Large model across 5 organ datasets (~400K slices).

**Outcome:** Failed the pre-registered gates. View retrieval peaked at 34× vs the 100×
gate; the model learned scanner identity (AUC 0.981) and spacing geometry over anatomy,
and the two smallest organs (colon, hepatic-vessel) collapsed into near-identical
embedding space (cosine 0.962).

**Why it matters:** proved that scaling model size does NOT fix a structurally-wrong
objective on multi-site medical data. This is what pushed us to organ specialists.

**Evidence:** ledger § Phase 5; `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`.

## 2. Invalid gates for single-organ models (view retrieval, spacing)

**Attempt:** Use view-retrieval ratio and spacing-prediction R² as quality gates for
single-organ specialists.

**Outcome:** Invalid. DINO's final-layer spatial-correspondence artifact handicaps view
retrieval on single-organ data (31→37×, never reaching 40×), and single-organ datasets
lack the spacing variance to drive scale embeddings (R² 0.941 < 0.95, counterfactual
0.239 < 0.30).

**Why it matters:** You cannot reuse pan-organ eval for specialists. Governance was changed
to gate specialists **solely on clinical utility (LoRA AUROC) + external validation**.

**Evidence:** ledger § Phase 7 (KILL), § Phase 8 (PIVOT).

## 3. Colon specialist — `[CONCLUDED — FAIL]`

**Attempt:** Apply the same organ-specialist recipe to MSD-Colon.

**Outcome:** LoRA AUROC **0.6529** — below the 0.70 gate, unstable training (AUROC
oscillated 0.58–0.65). Possible causes: small dataset (126 patients), a lung-tuned HU
window applied to colon, or colon texture being less distinctive for the scale-aware
ViT-Base. Not yet re-investigated.

**Why it matters:** honest boundary of the recipe — not every organ is equally
learnable with this approach. Also: a too-small validation set (10 patients) would have
been statistically useless, which is partly why we pursued external validation for the
winning model instead.

**Evidence:** ledger § Phase 9; `adapters/msd-colon-vit-base-50k/`.

## 4. The leakage finding (a negative result about our own process)

**Attempt:** Trust the internal AUROC 0.9456 as a release claim.

**Outcome:** Invalid — 93% of fine-tuning validation patients were in pretraining train
(representation leakage). The number was inflated.

**Why it matters:** It's the most important negative result here. It forced the external
salvage, and it added a documented failure mode (`Evaluation-Design Gates`) to the
governance skill so future runs (and other projects) check split disjointness by default.

**Evidence:** `docs/RED_TEAM_TIER1.md`; ledger external-validation entry discusses the
resolved concern.

---

### Golden rule
None of these were re-run blindly. Each points to a permanent record. If you're tempted
to "just try pan-organ again with more data," read the discontinuation notice first.
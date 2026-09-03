# 01 — Decision Trail: Why This Model Exists

This is the honest lineage of `dinox-ct-msd-hepatic-vessel-vit-base-v1`. It is not a
sanitized "big idea → success" story. It is the record of failures that made the
organ-specialist strategy — and this specific model — the right call.

---

## 1. The original bet: one model to rule all organs (`pan-organ`)

The initial DINO-X vision was a single foundation model that understood *all* organs.
We trained ViT-Small and ViT-Large on a 5-dataset corpus (LIDC, Pancreas-CT, MSD-Colon,
MSD-Hepatic-Vessel, CQ500 — ~400K slices).

**What we found (ledger, Phase 5, `[CONCLUDED — KILL]`):**
- View retrieval peaked at 34×, far below the 100× pre-registered gate.
- **Scanner fingerprinting:** the model learned to predict *which hospital* a scan came
  from (dataset discrimination AUC 0.981) instead of learning anatomy.
- **Capacity dilution:** the two smallest organs (colon, hepatic-vessel) collapsed into
  near-identical embedding space — the model abandoned them.

**The core lesson:** DINO + KoLeo on multi-site medical data optimizes for the *easiest*
signals (scanner identity, spacing geometry) over the *hardest* (cross-organ anatomy).
This was structural, not a hyperparameter problem — ViT-Small *and* ViT-Large hit the
same wall. Ledger: `docs/EXPERIMENTS_AND_RESULTS.md` § Phase 5; tree § `[CONCLUDED — KILL]`.
Artifacts: `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`.

## 2. The pivot: single-organ specialists

If a multi-organ model under pressure abandons small organs, then **one model per organ**
removes the dilution entirely. This became the governing strategy (Phase 6-9).

- A LIDC-only ViT-Base specialist reached LoRA AUROC **0.728** (project-best at the time).
- This model (`msd-hepatic-vessel`) was the second application of that proven recipe,
  alongside msd-colon.

## 3. The false ceiling: our own internal numbers looked too good

The hepatic-vessel internal LoRA AUROC came back **0.9456** — suspiciously high. Acting
as our own red-team (see `docs/RED_TEAM_TIER1.md`), we audited the split:

**Critical finding:** the pretraining used a *series-level* split (275 train / 28 val),
but the LoRA fine-tuning used an *independent* patient-level split. The result:

> **93% of the fine-tuning "held-out" validation patients were in the pretraining
> training set.** The backbone had already seen them during self-supervised pretraining,
> so the 0.9456 was inflated by representation leakage (memory of patient/scanner
> identity), not clean generalization.

This invalidated the internal AUROC as a *release* claim — the same class of error the
scanner-fingerprinting work warned about, subtler. Commit: `672454a` (red-team report).

## 4. The salvage: genuinely external validation

We could not un-see the leaked patients. The honest fix was a **genuinely external** test
set the model could not have memorized — a different institution, different scanners.

We chose **TCIA CRLM** (ColoRectal Liver Metastases): 197 subjects, multi-institution,
with SEG segmentations containing exactly the Hepatic + Portal vessel segments that
match the MSD Task08 formulation the model was trained on. CRLM cannot overlap MSD
training data by construction.

**Result (overview):** external slice AUROC **0.9413** on 17,639 slices from 197 new
patients. This resolved the leakage concern — the model transfers to unseen institutions
on correct vessel semantics. Detail: `the_science/03_validation.md`; ledger `d1aab1c`.

## 5. Why this trail matters

Every pivot above is evidence-backed and permanent in the ledger. A reader who distrusts
the model can start from the pan-organ *failure*, watch us catch our own leak, and verify
the external number themselves. That is the trust model.

---

### Evidence pointers
- Ledger: `docs/EXPERIMENTS_AND_RESULTS.md` (Phase 5 KILL, Phase 9 PIVOT, External Validation GO)
- Tree: `docs/EXPERIMENT_TREE.md`
- Red-team: `docs/RED_TEAM_TIER1.md`
- Pan-organ tombstone: `docs/DISCONTINUATION_NOTICE_5dataset-phase5-large-bs256-v2.md`
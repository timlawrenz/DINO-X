# Red-Team Report — DINO-X Tier 1 Models (msd-colon, msd-hepatic-vessel)

**Date:** 2026-08-31  
**Auditor:** Hermes Agent (red-team mode)  
**Scope:** Tier 1 release candidates — LIDC specialist (AUROC 0.728), hepatic-vessel specialist (AUROC 0.9456), colon specialist (AUROC 0.6529 FAIL)

---

## Executive Summary

The Tier 1 models have **critical methodological vulnerabilities** that undermine the claimed performance. The most severe issue is **representation leakage**: the backbone has seen 93–94% of "validation" patients during self-supervised pretraining, meaning the LoRA evaluation measures memorization-adjacent generalization, not true held-out performance. Additionally, the hepatic-vessel AUROC of 0.9456 is likely inflated by extreme slice-level label autocorrelation (99.5% of positive slices are adjacent to another positive slice).

**Recommendation:** Do NOT release Tier 1 models with current evaluation results. The AUROC numbers are not trustworthy for external claims. Re-run evaluation with properly disjoint pretrain/finetune splits, or reframe claims as "internal validation" with explicit leakage disclosure.

---

## Critical Findings (Release Blockers)

### 1. REPRESENTATION LEAKAGE — Backbone Has Seen "Held-Out" Patients

**Severity: CRITICAL — invalidates AUROC claims**

The pretraining uses a series-level split (116 train / 10 val for colon, 275/28 for hepatic-vessel). The LoRA fine-tuning uses a **different, independent patient-level split** (70/15/15, seed=42). These splits are not aligned.

| Dataset | Finetune val patients in pretrain train | Finetune val patients in pretrain val |
|---|---|---|
| msd-colon | **17/18 (94%)** | 1/18 |
| msd-hepatic-vessel | **41/44 (93%)** | 3/44 |

**What this means:** The backbone was trained (self-supervised) on slices from patients that are later used to evaluate LoRA fine-tuning. The backbone has already learned patient-specific features (scanner noise, anatomy quirks, tissue textures) during pretraining. When LoRA fine-tuning achieves AUROC 0.9456 on these "held-out" patients, it's not measuring generalization to unseen patients — it's measuring how well the backbone's existing patient-specific representations transfer to the downstream task.

**Why this is bad:**
- The AUROC is inflated by an unknown amount (likely 0.02–0.10 based on similar leakage studies)
- External users cannot reproduce these numbers on truly held-out data
- The "clinical utility" claim is overstated — real-world deployment would see entirely new patients
- This is the same class of error as the scanner fingerprinting problem that killed pan-organ, but subtler

**Fix:** Re-run LoRA evaluation using only patients from the pretraining VAL split (10 colon patients, 28 hepatic-vessel patients). This gives a true held-out estimate. Alternatively, re-pretrain with a 3-way split (pretrain train / pretrain val / finetune test) to ensure disjointness.

---

### 2. SLICE-LEVEL LABEL AUTOCORRELATION — AUROC Is Not Slice-Independent

**Severity: HIGH — inflates hepatic-vessel AUROC**

99.5% of positive slices in the hepatic-vessel dataset are adjacent to another positive slice. The model doesn't need to learn "is there a tumor?" — it needs to learn "is this slice similar to the previous slice?" Because the backbone was pretrained with 3-slice context (z-1, z, z+1), and LoRA fine-tuning uses single slices, the model can exploit temporal adjacency: if slice N has tumor, slice N+1 almost certainly does too.

**Evidence:**
- Hepatic-vessel: 99.5% of positive-positive pairs are adjacent slices
- Median tumor pixels per positive slice: 695 (small tumors spread across many slices)
- The task is effectively "detect the tumor track," not "detect tumor presence"

**Why this inflates AUROC:**
- The model can achieve high AUROC by learning smooth features that vary slowly across z
- Random shuffling of slices during training breaks this correlation, but the evaluation metric (AUROC on individual slices) doesn't account for it
- A model that just predicts "previous slice's label" would achieve high AUROC on this data

**Fix:** Report patient-level AUROC (aggregate all slices per patient, then compute AUROC on patient-level scores). This is the clinically meaningful metric — a radiologist doesn't need 47 slice-level predictions, they need one patient-level answer.

---

### 3. NEGATIVE SLICE SAMPLING IS NOT STRATIFIED — Class Balance Varies Wildly by Patient

**Severity: MEDIUM — affects model calibration and fairness**

The `negative_ratio=1.0` in `extract_msd_labels.py` attempts to sample 1 negative per positive per patient, but is capped by available negative slices. This creates severe within-patient imbalance:

| Metric | Value |
|---|---|
| Patients with neg/pos ratio < 0.5 | 88/303 (29%) |
| Minimum neg/pos ratio | 0.09 (1 negative for 11 positives) |
| Median neg/pos ratio | 0.68 |

**Why this matters:**
- Patients with many positive slices and few negatives contribute disproportionately to the positive class
- The model may learn patient-specific biases rather than general tumor features
- Calibration varies by patient — some patients are "easy" (many negatives, clear baseline), others are "hard" (mostly positives)

**Fix:** Use global stratified sampling instead of per-patient sampling. Pool all negative slices across patients, then sample to achieve global 50/50 balance. Or use weighted loss to account for per-patient imbalance.

---

### 4. NO TEST SET EVALUATION — The Test Sets Exist But Were Never Used

**Severity: MEDIUM — incomplete evaluation**

Both datasets have properly held-out test sets:
- msd-colon: 394 slices, 20 patients
- msd-hepatic-vessel: 3,172 slices, 47 patients

The LoRA evaluation only reports validation AUROC. The test set was never touched. This means:
- We don't know the true generalization gap (val → test)
- We can't detect overfitting to the validation set via early stopping
- The reported AUROC is optimistic (best epoch selected on val)

**Fix:** Run the best checkpoint on the test set. Report val AUROC and test AUROC separately. If test AUROC drops significantly (>0.02), the model is overfitting to the validation set via early stopping.

---

### 5. COLON FAILURE IS NOT INVESTIGATED — Just Marked FAIL and Moved On

**Severity: MEDIUM — missed learning opportunity**

The colon specialist achieved AUROC 0.6529, failing the 0.70 gate. The ledger notes possible causes but doesn't investigate:
- Was the HU window (-30/120, lung-tuned) inappropriate for colon?
- Did the small dataset (126 patients) cause overfitting?
- Is colon CT fundamentally harder for scale-aware ViT-Base?
- Was the label extraction correct? (tumor_pixels distribution looks reasonable, but we didn't visually verify)

**Missing tests:**
- Visual inspection: Do the extracted labels match actual tumor locations?
- HU window ablation: Try colon-specific window (e.g., level=50, width=400)
- Data augmentation: The colon dataset is small — did we use sufficient augmentation?
- Architecture ablation: Would ViT-Small (fewer params) work better on this small dataset?

**Fix:** Before re-running colon, do a failure analysis: visualize misclassified slices, check HU window appropriateness, verify label quality.

---

## High-Priority Findings (Should Fix Before Release)

### 6. SINGLE SEED EVALUATION — No Reproducibility Check

All LoRA evaluations used seed=42. The ablation scan (Phase 4) showed run-to-run variance of ~0.016 AUROC on LIDC. For the hepatic-vessel AUROC of 0.9456, a ±0.016 confidence interval means the true value could be 0.93–0.96. For the colon FAIL at 0.6529, the true value could be 0.64–0.67 — still FAIL, but the margin matters for borderline cases.

**Fix:** Run 3 seeds (42, 123, 456) and report mean ± std. If the std is large (>0.02), the result is not stable enough for release.

### 7. NO CALIBRATION METRICS — AUROC Alone Is Insufficient for Clinical Use

The evaluation reports AUROC, accuracy, and macro-F1, but not calibration metrics. For clinical deployment, a model that says "90% confidence" should be right 90% of the time. AUROC doesn't measure this.

**Missing metrics:**
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagram

**Fix:** Add calibration evaluation to `finetune_lora.py`. The MC Dropout paper (arXiv:2607.16317) showed that deterministic confidence heads collapse — use MC Dropout or temperature scaling for calibrated probabilities.

### 8. NO EXTERNAL VALIDATION — Zero-Shot Transfer Never Tested

The UKBOB paper (arXiv:2504.06908) validates their model by zero-shot transfer to external datasets (AMOS, BTCV). DINO-X has no equivalent test. The hepatic-vessel specialist achieves 0.9456 on MSD data, but would it work on a different hospital's CT scans?

**Missing tests:**
- Zero-shot transfer: Train on MSD hepatic-vessel, test on a different liver CT dataset
- Scanner robustness: Does the model work on GE vs Siemens vs Philips scanners?
- Resolution robustness: Does the model work on 1mm slices vs 5mm slices?

**Fix:** Identify at least one external liver CT dataset and run zero-shot evaluation. If AUROC drops significantly, the model is overfitting to MSD's specific acquisition protocol.

### 9. ADVERSARIAL PASS CHECKLIST IS INCOMPLETE — All 4 Questions Marked ❌ or N/A

The Phase 9 ledger entry has an adversarial pass checklist with all items unchecked or marked N/A. This violates the project's own governance rule: "never write PASS in the ledger until this checklist is complete."

Current status:
- ❌ Metric code has unit tests — No test suite for AUROC computation
- ✅ Metric definition unchanged — Same LoRA protocol
- ❌ Result reproduced — Single seed=42 run
- ❌ Extremes + edge cases inspected — Not performed

**Fix:** Complete the adversarial pass before claiming PASS. At minimum: (1) add unit tests for `_compute_auroc`, (2) run multi-seed, (3) visually inspect top/bottom predictions.

---

## Medium-Priority Findings (Improve Quality)

### 10. WINDOW HARDCODING — Lung Window Used for All Organs

The LoRA fine-tuning uses `--window-level -30 --window-width 120` (lung window) for both colon and hepatic-vessel. This is anatomically inappropriate:
- Colon soft tissue: level ~50, width ~400
- Liver: level ~100, width ~200

Using a lung window for abdominal organs clips the relevant HU range, potentially destroying discriminative signal.

**Fix:** Add organ-specific window presets to `finetune_lora.py`. At minimum, document why the lung window was chosen for abdominal organs.

### 11. NO CROP SIZE ABLATION FOR MSD DATASETS

LIDC experiments showed 64px >> 128px >> 224px for nodule classification. The MSD evaluations use the default 224px (backbone's native resolution). But colon tumors and hepatic vessels may have different optimal crop sizes.

**Fix:** Run crop size ablation for hepatic-vessel (64px, 128px, 224px). If 64px works better, the current 224px result may be suboptimal.

### 12. NO COMPARISON TO BASELINE — "Clinically Useful" Is Unsubstantiated

The 0.70 AUROC gate was chosen as "clinically useful" without justification. For context:
- Random: 0.50
- LIDC specialist: 0.728
- Hepatic-vessel: 0.9456
- Published liver tumor detection: typically 0.90–0.95

0.9456 is competitive with published results, but 0.728 is marginal. The gate should be justified per-organ based on clinical literature, not set uniformly.

**Fix:** Document the clinical rationale for each organ's AUROC gate. For colon, 0.70 may be too low (published colon cancer detection is typically 0.85+). For hepatic-vessel, 0.90 may be more appropriate.

### 13. NO ERROR ANALYSIS — We Don't Know What the Model Gets Wrong

The evaluation reports aggregate metrics but doesn't analyze failure modes:
- Are false positives near tumors (boundary cases) or in completely normal tissue?
- Are false negatives small tumors or large ones?
- Do errors cluster by patient, scanner, or slice position?

**Fix:** Add error analysis to the evaluation pipeline. Visualize the top false positives and false negatives. Check if errors correlate with tumor size, slice position, or patient characteristics.

### 14. TRAINING INSTABILITY IN COLON — Not Investigated

The colon LoRA training showed AUROC oscillating 0.58–0.65 across epochs. This is unusual — LIDC and hepatic-vessel both showed smooth convergence. The instability suggests:
- Learning rate too high for this dataset
- Data quality issues (corrupted slices, wrong labels)
- Class imbalance within batches
- Backbone features not suited to colon texture

**Fix:** Investigate before re-running. Check for corrupted slices (SOLF-like filters), try lower LR, try ViT-Small backbone.

---

## Low-Priority Findings (Nice to Have)

### 15. NO MODEL CARD — Release Readiness Incomplete

The adapters have placeholder READMEs but no proper HF model cards. A model card should include: intended use, limitations, training data, evaluation results, ethical considerations, and citation.

### 16. NO UNCERTAINTY QUANTIFICATION

The MC Dropout paper showed that uncertainty estimation is critical for clinical deployment. DINO-X has no uncertainty mechanism. Adding MC Dropout to LoRA would enable "defer to human" for uncertain cases.

### 17. NO EFFICIENCY METRICS

The models are evaluated on accuracy but not efficiency. For clinical deployment, inference time and memory matter. The Xray-Visual paper showed 4× efficiency gains from token reduction — DINO-X could benefit from similar optimizations.

---

## Summary Table

| # | Finding | Severity | Release Blocker? | Fix Effort |
|---|---|---|---|---|
| 1 | Representation leakage (93–94% overlap) | CRITICAL | YES | Medium (re-split) |
| 2 | Slice-level autocorrelation (99.5%) | HIGH | YES | Low (patient-level AUROC) |
| 3 | Non-stratified negative sampling | MEDIUM | No | Low |
| 4 | No test set evaluation | MEDIUM | YES | Low (run test) |
| 5 | Colon failure not investigated | MEDIUM | No | Medium |
| 6 | Single seed evaluation | HIGH | YES | Low (3 seeds) |
| 7 | No calibration metrics | HIGH | No | Medium |
| 8 | No external validation | HIGH | YES | Medium |
| 9 | Adversarial pass incomplete | HIGH | YES | Medium |
| 10 | Wrong HU window for organs | MEDIUM | No | Low |
| 11 | No crop size ablation | MEDIUM | No | Low |
| 12 | No baseline justification | LOW | No | Low |
| 13 | No error analysis | MEDIUM | No | Medium |
| 14 | Colon instability not investigated | MEDIUM | No | Medium |
| 15 | No model card | LOW | No | Low |
| 16 | No uncertainty quantification | LOW | No | Medium |
| 17 | No efficiency metrics | LOW | No | Medium |

---

## Recommended Actions Before Release

### Immediate (Block Release)
1. **Fix representation leakage** — Re-run LoRA evaluation using only pretrain-val patients, or re-pretrain with 3-way split
2. **Report patient-level AUROC** — Aggregate slice predictions to patient level before computing AUROC
3. **Run test set evaluation** — Report val and test AUROC separately
4. **Multi-seed evaluation** — 3 seeds, report mean ± std

### Short-term (Before Announcement)
5. **Complete adversarial pass** — Unit tests, visual inspection, edge cases
6. **External validation** — Zero-shot transfer to at least one external dataset
7. **Calibration metrics** — ECE, Brier score, reliability diagram
8. **Investigate colon failure** — Visual inspection, HU window ablation

### Medium-term (Post-Release)
9. **Patient-level evaluation** — Replace slice-level AUROC with patient-level
10. **Organ-specific windows** — Add per-organ HU window presets
11. **Error analysis pipeline** — Automated failure mode detection
12. **Uncertainty quantification** — MC Dropout for clinical gating

---

*This red-team report follows the adversarial pass methodology in `docs/experiment-structure.md`. All findings are documented in the permanent ledger at `docs/EXPERIMENTS_AND_RESULTS.md`.*

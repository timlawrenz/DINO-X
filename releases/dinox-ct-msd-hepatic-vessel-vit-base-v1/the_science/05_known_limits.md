# 05 — Known Limitations

Honest scope of this model. Where it is validated, where it is not.

---

## Validated
- **Hepatic/portal vessel tissue detection in contrast-enhanced abdominal CT** — slice-level,
  internal (MSD, AUROC 0.9456) and external (TCIA CRLM, AUROC 0.9413).
- **Transfer across institutions** on this specific vessel-detection task.

## Known limitations / not validated

1. **Not a clinical diagnostic device.** Research use only. Do not use for patient
   diagnosis, treatment planning, or any clinical decision. AUROC ≠ clinical safety;
   calibration and error analysis for clinical deployment have not been performed.

2. **Single modality (CT), single task (vessel-present per slice).** Not validated on
   MRI, ultrasound, or other organs. Do not assume the backbone generalizes beyond
   abdominal CT vessel/soft-tissue detection.

3. **Slice-level, not a full segmentation or 3D diagnostic.** The model classifies a 2D
   axial slice as "vessel tissue present/absent"; it does not delineate vessels or reason
   across the volume in 3D. The pretrained backbone used 3-slice context; the fine-tuned
   eval is single-slice (a documented distribution shift).

   **Single-slice path is validated, not a shortcut.** The pretrained backbone learned
   with 3-slice context `[z−1, z, z+1]`, while the released `predict_proba` duplicates a
   single slice `[z, z, z]`. We measured the cost of this directly on the external CRLM
   set (no retraining): true 3-slice context scored AUROC **0.94113** vs duplicated
   **0.94130** (Δ = −0.0002, 17,639 slices). LoRA fine-tuning adapted the head to the
   duplicated-slice distribution, so discarding spatial context costs nothing measurable
   on this task. See `docs/EXPERIMENT_TREE.md` § 3-slice context ablation.

4. **Internal metrics carry a pretraining-exposure caveat.** The internal 0.9456 was
   measured on fine-tuning patients the backbone had partly seen (93% of the fine-tune
   *validation* patients were in the pretraining corpus). Because pretraining used all
   48,021 MSD Hepatic-Vessel slices, **the internal test set (47 patients) was likewise
   entirely present** in the self-supervised pretraining data. Internal numbers (val and
   test) are reported for transparency only; the *external* CRLM 0.9413 is the
   release-supporting number. Do not cite internal-only numbers as generalization.

   **Same-dataset design (stated plainly).** This is a single-organ *specialist*: the
   backbone was self-supervised on MSD Hepatic-Vessel and the LoRA task head was then
   fine-tuned on labeled slices from the *same* MSD Hepatic-Vessel dataset. That is
   intentional — the design goal is organ-specific feature quality, not cross-dataset
   pretraining diversity — but it means the task head was optimized on data distribution
   the backbone was exposed to during pretraining. The defense against this being a
   memorization artifact is the **external CRLM evaluation**: a different institution,
   different scanners, zero patient overlap, where the model still achieves 0.9413. If
   you need a model whose fine-tuning distribution is disjoint from pretraining, this is
   not that model.

5. **Single acquisition protocol lineage.** Both MSD and CRLM are contrast-enhanced
   abdominal CT. Non-contrast CT, unusual HU windows, or norm windowing may degrade
   performance. We applied a narrow soft-tissue/vascular window (level −30, width 120,
   i.e. the [−90, +30] HU range) to match pretraining; this may not be optimal for all
   downstream tasks.

6. **No calibrated probabilities / no uncertainty.** Outputs are softmax probabilities
   without temperature scaling or MC-Dropout uncertainty. For any calibrated-confidence
   or defer-to-human use case, calibration must be added.

7. **Patient-level inference not fully solved.** A patient-level ROC-AUC is undefined on
   the external set (196/197 patients vessel-positive, 1 all-negative), so we report the
   Spearman rank correlation (0.527) instead. The ideal patient-level aggregation for
   clinical use (e.g. per-patient decision from slice scores) is not yet established.

---

## Scope of the external validation
The external result is strong **for the evaluated distribution** (TCIA CRLM vessel
tissue). It is not proof of universal robustness; treat the 17,639-slice / 197-patient
external set as the defensible generalization claim, not an upper bound.
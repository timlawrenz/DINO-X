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

4. **Internal AUROC carries leakage caveat.** The internal 0.9456 was measured on fine-tuning
   patients the backbone had partly seen. It is reported for transparency; the *external*
   0.9413 is the release-supporting number. Do not cite internal-only numbers as
   generalization.

5. **Single acquisition protocol lineage.** Both MSD and CRLM are contrast-enhanced
   abdominal CT. Non-contrast CT, unusual HU windows, or norm windowing may degrade
   performance. We applied a lung window (level -30, width 120) to match pretraining;
   this may not be optimal for all downstream tasks.

6. **No calibrated probabilities / no uncertainty.** Outputs are softmax probabilities
   without temperature scaling or MC-Dropout uncertainty. For any calibrated-confidence
   or defer-to-human use case, calibration must be added.

7. **Patient-level inference not fully solved.** External patient-level AUROC (0.739,
   majority-vote) is meaningful but the ideal patient-level aggregation for clinical use
   (e.g. per-patient decision from slice scores) is not yet established.

---

## Scope of the external validation
The external result is strong **for the evaluated distribution** (TCIA CRLM vessel
tissue). It is not proof of universal robustness; treat the 17,639-slice / 197-patient
external set as the defensible generalization claim, not an upper bound.
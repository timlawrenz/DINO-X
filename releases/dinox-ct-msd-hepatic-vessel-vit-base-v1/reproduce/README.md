# Reproduce The Headline Numbers

Every headline number can be recomputed from this envelope. The script refs expect the
artifacts on the same machine layout (or cloned repo + NAS paths). Exact commands below.

---

## Repro 1 — External slice-level AUROC (0.9413) [RELEASE-SUPPORTING]

Needs: the CRLM preprocessed slices + per-slice vessel labels (on NAS) and the adapter.

```bash
cd /home/tim/activity/DINO-X
source scripts/rocm_env.sh

# Generate vessel labels from CRLM SEG (if not already built)
python scripts/preprocessing/extract_crlm_vessel_labels.py \
  --seg-root /mnt/nas-ai-models/training-data/dino-x/crlm/seg \
  --ct-root  /mnt/nas-ai-models/training-data/dino-x/crlm/ct \
  --index-csv /mnt/nas-ai-models/training-data/dino-x/crlm/processed/crlm/index.csv \
  --output data/crlm/vessel_labels.csv

# Run the external eval (slice + patient AUROC). Stage slices to local NVMe for speed:
python scripts/eval_external.py \
  --adapter adapters/msd-hepatic-vessel-vit-base-50k \
  --label-csv data/crlm/vessel_labels_local.csv \
  --window-level -30 --window-width 120 \
  --device cuda
# Expect: Slice-level AUROC 0.9413, Spearman 0.527 (p~1e-15)
#   (patient-level ROC-AUC is undefined: 196/197 patients vessel-positive, 1 negative)
```

`scripts/eval_external.py` persists per-slice predictions to
`adapters/.../external_probs_*.npz` immediately after inference, so even if an
aggregation step fails you keep the raw predictions.

## Repro 2 — Internal LoRA AUROC (0.9456, leakage-caveated)

```bash
python scripts/finetune_lora.py \
  --backbone runs/20260728_180949_hepatic-specialist-vit-base-scale-aware/checkpoint_final_00050000.pth \
  --train-csv data/msd-hepatic-vessel/labels/msd_hepatic_vessel_train.csv \
  --val-csv   data/msd-hepatic-vessel/labels/msd_hepatic_vessel_val.csv \
  --data-root /home/tim/activity/DINO-X \
  --window-level -30 --window-width 120 --num-classes 2 \
  --rank 8 --alpha 16 --epochs 50 --batch-size 32 --lr 5e-4 \
  --es-metric auroc --patience 10 --seed 42 \
  --output adapters/msd-hepatic-vessel-vit-base-50k
```
> **Do not cite this number as generalization** — 93% of these validation patients were in
> pretraining. It is for transparency. The external 0.9413 is the release number.

## Repro 3 — Leakage audit (why external validation was needed)
```bash
# Check how many fine-tune val patients overlap pretrain train
python - <<'EOF'
import json
m = json.load(open("data/mvp/split_manifest_msd_hepatic_vessel.json"))
val_series = set(m["val"]["series_dir"])
# (mapping: fine-tune val patients -> their series, then intersect with val_series)
# Result documented: 41/44 (93%) in pretrain train
EOF
```

---

## Envelope integrity
`reproduce/repro_metrics.py` asserts the persisted predictions match the claimed metrics
and that every `evaluation.json` field has a corresponding artifact.
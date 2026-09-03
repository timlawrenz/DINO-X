# Data Catalog & Provenance

**Model:** `dinox-ct-msd-hepatic-vessel-vit-base-v1`  

**Git commit:** `9b34237e4d0046f55a83748e5acd817db609325a`  

**Dataset catalog SHA-256 (16):** `7eebf9085200017b`


> Generated from the actual on-disk index/label files, not hand-typed counts. Machine-readable twin: `data_provenance.json`.


## 1. Datasets Involved

| Dataset | Source | License | Redistribution | Slices-in-index |
|---|---|---|---|---|
| msd-hepatic-vessel | http://medicaldecathlon.com/ | CC-BY-SA-4.0 | allowed_with_conditions | 48021 |

## 2. Pretraining Training Index

- **Path:** `data/mvp/msd_hepatic_vessel_only_t2.csv`
- **Content SHA-256 (16):** `71079e567d7fcb33`
- **Total slices:** 48021
  - `msd-hepatic-vessel`: 48021 slices

**Spacing distribution:**
```json
{
  "spacing_x": {
    "min": 0.5684,
    "max": 0.9766,
    "mean": 0.8026
  },
  "spacing_y": {
    "min": 0.5684,
    "max": 0.9766,
    "mean": 0.8026
  },
  "slice_thickness": {
    "min": 0.8,
    "max": 8.0,
    "mean": 3.2779
  }
}
```

**Split manifest:**
```json
{
  "train": {
    "count": 275,
    "sample": [
      "msd-hepatic-vessel_hepaticvessel_001",
      "msd-hepatic-vessel_hepaticvessel_002",
      "msd-hepatic-vessel_hepaticvessel_004"
    ]
  },
  "val": {
    "count": 28,
    "sample": [
      "msd-hepatic-vessel_hepaticvessel_007",
      "msd-hepatic-vessel_hepaticvessel_013",
      "msd-hepatic-vessel_hepaticvessel_026"
    ]
  },
  "counts": {
    "train_series": 275,
    "val_series": 28
  }
}
```

## 3. Fine-Tuning Labels


**finetune_train** (`data/msd-hepatic-vessel/labels/msd_hepatic_vessel_train.csv` hash `3b6403427a585457`):
- rows: 14229, positive: 8591 (0.6038), negative: 5638, patients: 212

**finetune_val** (`data/msd-hepatic-vessel/labels/msd_hepatic_vessel_val.csv` hash `335cc4dbda9f2b25`):
- rows: 2840, positive: 1790 (0.6303), negative: 1050, patients: 44

**test** (`data/msd-hepatic-vessel/labels/msd_hepatic_vessel_test.csv` hash `6dcc72579cb2685e`):
- rows: 3172, positive: 1962 (0.6185), negative: 1210, patients: 47

## 4. External Validation

- **Path:** `data/crlm/vessel_labels.csv` hash `37651fb38ce10660`
- **Slices:** 17639, patients: 197, positive: 8992 (0.5098)
- **Note:** External held-out set, disjoint from training (see the_science/03_validation.md)

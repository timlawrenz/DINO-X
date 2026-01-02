# Phase 2: Data (Fuel)

This project keeps **raw medical data out of the Git repo**.
In our environment, the raw dataset lives on a NAS mounted at:

- `/mnt/nas-ai-models/`

## Directory Layout

Recommended external layout:

```
/mnt/nas-ai-models/
└── training-data/
    └── dino-x/
        └── lidc-idri/
            ├── raw/                 # DICOMs as acquired (never committed)
            └── processed-2p5d-rgb/   # Derived PNGs + index (also never committed)
```

Repository-local (safe) paths:

```
./data/
├── raw/        -> /mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw
└── processed/  -> /mnt/nas-ai-models/training-data/dino-x/lidc-idri/processed-2p5d-rgb
```

## Environment Variables

- `DINOX_DATA_ROOT`: overrides the default external root.
  - Default: `/mnt/nas-ai-models/training-data/dino-x`

## Bootstrap the symlinks

Idempotently create the `data/` symlinks:

```bash
bash scripts/phase2_setup_data_root.sh
```

To use a non-NAS location:

```bash
DINOX_DATA_ROOT=/path/to/training-data/dino-x bash scripts/phase2_setup_data_root.sh
```

## Acquire LIDC-IDRI (TCIA)

Use the TCIA REST helper to download **CT** series (avoid DX series with only a few images):

```bash
python3 scripts/phase2_tcia_download.py list-series \
  --collection LIDC-IDRI \
  --modality CT \
  --min-image-count 50 \
  --sort-by imagecount \
  --out /tmp/lidc_ct_series_uids.txt

python3 scripts/phase2_tcia_download.py download-series \
  --uids /tmp/lidc_ct_series_uids.txt \
  --out-root data/raw
```

## Preprocess LIDC-IDRI (DICOM -> 2.5D PNG)

```bash
python3 scripts/phase2_preprocess_lidc_idri.py \
  --dicom-root data/raw \
  --out-root data/processed
```

## Visual validation

```bash
python3 scripts/phase2_validate_samples.py \
  --processed-root data/processed \
  --out-dir data/processed/_validation
```

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
            └── processed-hu16/       # Derived 16-bit HU PNG slices + index (also never committed)
```

Repository-local (safe) paths:

```
./data/
├── raw/        -> /mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw
└── processed/  -> /mnt/nas-ai-models/training-data/dino-x/lidc-idri/processed-hu16
```

## Environment Variables

- `DINOX_DATA_ROOT`: overrides the default external root.
  - Default: `/mnt/nas-ai-models/training-data/dino-x`

## Why the order matters

We’re building a **reproducible dataset state** from (large) external raw data:

1. **Stable local paths** (`data/raw`, `data/processed`) come first so every subsequent step is predictable.
2. **Acquisition** must happen before preprocessing (obvious), but also we must **filter to CT series**; TCIA returns multiple modalities and some series have only 1–2 images (not usable for 2.5D).
3. **Preprocessing** is expensive and produces lots of files; we only want to run it once we’ve confirmed acquisition is correct.
4. **Validation + manifest** are your “receipt”: they prove the data looks right and capture counts/sizes so you can detect partial downloads later.

## Phase 2 runbook (copy/paste)

### 1) Bootstrap the symlinks (safe local paths)

Idempotently create the `data/` symlinks:

```bash
bash scripts/phase2_setup_data_root.sh
```

To use a non-NAS location:

```bash
DINOX_DATA_ROOT=/path/to/training-data/dino-x bash scripts/phase2_setup_data_root.sh
```

**Why:** keeps raw/derived data off-repo, but gives scripts a stable local path.

### 2) Acquire LIDC-IDRI (TCIA, CT-only)

(Optional) Generate a list of **CT** series UIDs with enough slices to support 2.5D:

```bash
python3 scripts/phase2_tcia_download.py list-series \
  --collection LIDC-IDRI \
  --modality CT \
  --min-image-count 50 \
  --sort-by imagecount \
  --out /tmp/lidc_ct_series_uids.txt
```

If you already have a UID list, you can download just those with:

```bash
python3 scripts/phase2_tcia_download.py download-series \
  --uids /tmp/lidc_ct_series_uids.txt \
  --out-root data/raw
```

Download + extract series into `data/raw` (run under `tmux` for long downloads).
This is **resumable by default**: re-run after an interruption.

```bash
python3 scripts/phase2_tcia_download.py download-collection \
  --collection LIDC-IDRI \
  --modality CT \
  --min-image-count 50 \
  --sort-by imagecount \
  --out-root data/raw \
  --out-uids /tmp/lidc_ct_series_uids.txt
```

**Why:** the NBIA `getSeries` endpoint returns other modalities (e.g. DX) that often have only 1–2 images; those will be skipped by preprocessing.

### 3) Preprocess (DICOM -> 16-bit HU PNG slices)

```bash
python3 scripts/phase2_preprocess_lidc_idri.py \
  --dicom-root data/raw \
  --out-root data/processed
```

**What it produces:**
- PNGs under `data/processed/lidc-idri/...`
- An index CSV at `data/processed/_index/index.csv`

### 4) Visual validation (spot-check output quality)

```bash
python3 scripts/phase2_validate_samples.py \
  --processed-root data/processed \
  --out-dir data/processed/_validation_$(date +%Y%m%d_%H%M%S) \
  --seed $(date +%s)
```

**Why:** confirms slices decode correctly and a fixed preview window looks sane before you start any training.

### 5) Write a provenance manifest (counts + bytes)

```bash
python3 scripts/phase2_write_dataset_manifest.py \
  --dataset lidc-idri \
  --raw-root data/raw \
  --processed-root data/processed \
  --out data/processed/_manifests/lidc-idri.manifest.json
```

**Why:** records a simple, non-PHI “dataset receipt” so Phase 3 runs can reference a specific dataset state.

## Common gotchas

- `data/raw` and `data/processed` are symlinks; `find` won’t follow them unless you use `find -L ...`.
- If you see `skip=true ... reason=too_few_slices`, you likely downloaded a non-CT series or a CT series with too few images.

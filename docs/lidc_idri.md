# LIDC-IDRI Acquisition (Raw Data)

LIDC-IDRI is distributed via **TCIA (The Cancer Imaging Archive)** and is ~120GB.
Because this is medical imaging data, **do not commit it to Git**; store it in the NAS-backed raw data folder.

## Target raw layout

Place the acquired files under:

- `/mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw/`

Or set `DINOX_DATA_ROOT` and use the symlink bootstrap helper:

```bash
DINOX_DATA_ROOT=/path/to/training-data/dino-x bash scripts/phase2_setup_data_root.sh
```

After bootstrapping, `data/raw` should point at the external raw directory.

## Programmatic acquisition (TCIA REST API)

TCIA exposes NBIA REST endpoints to (1) list SeriesInstanceUIDs in a collection and (2) download a series as a ZIP.
This repo includes a small helper:

- `scripts/phase2_tcia_download.py`

### 1) List series UIDs for LIDC-IDRI (optional)

Note: `getSeries` returns multiple modalities; for preprocessing we want **CT** series with many slices.

```bash
# Filter to CT series and prefer series with many images (slices).
python3 scripts/phase2_tcia_download.py list-series \
  --collection LIDC-IDRI \
  --modality CT \
  --min-image-count 50 \
  --sort-by imagecount \
  --out /tmp/lidc_ct_series_uids.txt
```

### 2) Download/extract *all* CT series into `data/raw/` (recommended)

This resolves the series list and downloads it in one step.
It is **resumable by default**: re-run the same command after an interruption.

```bash
python3 scripts/phase2_tcia_download.py download-collection \
  --collection LIDC-IDRI \
  --modality CT \
  --min-image-count 50 \
  --sort-by imagecount \
  --out-root data/raw \
  --out-uids /tmp/lidc_ct_series_uids.txt
```

(If you want the old two-step flow, you can still use `list-series` + `download-series`.)

Optional auth:
- Public collections often work without a key.
- If required, set `TCIA_API_KEY` (or pass `--api-key ...`).

Example:

```bash
export TCIA_API_KEY=...  # if needed
python3 scripts/phase2_tcia_download.py list-series --collection LIDC-IDRI --modality CT --min-image-count 50 --sort-by imagecount --out /tmp/lidc_ct_series_uids.txt
```

## Provenance / manifest

After acquisition, write a minimal manifest JSON (counts + total bytes) for traceability:

```bash
python3 scripts/phase2_write_dataset_manifest.py \
  --dataset lidc-idri \
  --raw-root data/raw \
  --processed-root data/processed \
  --out data/processed/_manifests/lidc-idri.manifest.json
```

Notes:
- This manifest is intended for reproducibility metadata only; it should not include any patient-identifying fields.

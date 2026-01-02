## 1. Implementation
- [x] 1.1 Document dataset storage conventions (NAS mount, local symlink pattern, environment variables) in `docs/`.
- [x] 1.2 Add repo safety defaults to prevent committing data (e.g., ignore `data/`, DICOM, PNG/JPEG outputs).
- [x] 1.3 Implement a data-root bootstrap helper (script) that creates idempotent symlinks (e.g., `data/raw` → `/mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw`).
- [x] 1.4 Implement LIDC-IDRI acquisition instructions and manifest tracking (source URL, expected size, optional checksums).
- [x] 1.5 Implement preprocessing pipeline: DICOM → HU normalization → 2.5D stacking → RGB windowing (Lung/Soft-Tissue/Bone) → PNG output + index/manifest.
- [x] 1.6 Implement validation command to generate 10 random sample images plus a single contact sheet for quick inspection.
- [x] 1.7 Add a small “dry-run” mode using a tiny synthetic/fixture volume so preprocessing logic can be validated without the real dataset.

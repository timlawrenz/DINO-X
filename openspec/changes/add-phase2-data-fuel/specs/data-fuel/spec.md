## ADDED Requirements

### Requirement: External Raw Data Root (NAS-backed)
The system SHALL support storing raw training data outside the Git repository.

#### Scenario: NAS mount is used for raw data
- **WHEN** the host has the NAS mounted at `/mnt/nas-ai-models/`
- **THEN** raw datasets SHALL be stored under a project-specific subfolder (e.g., `/mnt/nas-ai-models/training-data/dino-x/`)
- **AND** the repository SHALL reference raw data via a local symlink (e.g., `data/raw` → `/mnt/nas-ai-models/training-data/dino-x/lidc-idri/raw`) to reduce risk of accidental commits.

#### Scenario: Alternate raw data root is used
- **WHEN** the NAS mount is not available
- **THEN** the user SHALL be able to configure an alternate raw data root via an environment variable (e.g., `DINOX_DATA_ROOT`)
- **AND** all Phase 2 scripts SHALL use that configured root instead of hard-coding `/mnt/nas-ai-models/`.

### Requirement: Repository Safety for Medical Data
The system SHALL provide safeguards to prevent raw medical data and large derived datasets from being committed to Git.

#### Scenario: Data folders are ignored by default
- **WHEN** the repository is cloned
- **THEN** default ignore rules SHALL exclude data directories and common medical imaging formats (e.g., DICOM) and derived image outputs (e.g., PNG)
- **AND** documented workflow SHALL rely on external storage and symlinks rather than in-repo copies of raw datasets.

### Requirement: LIDC-IDRI Acquisition and Provenance
The system SHALL define how to acquire the LIDC-IDRI dataset and record provenance sufficient to reproduce the dataset state.

#### Scenario: Dataset acquisition is documented and traceable
- **WHEN** a user prepares Phase 2 data
- **THEN** documentation SHALL specify the acquisition source (TCIA) and the expected dataset scale (~120GB)
- **AND** the system SHALL record a dataset manifest (at minimum: dataset name, acquisition date, and directory layout; optionally: checksums).

### Requirement: Deterministic HU16 Preprocessing Pipeline
The system SHALL provide a preprocessing pipeline that converts CT DICOM volumes into **16-bit lossless HU slice PNGs** (one file per axial slice), preserving raw density values.

#### Scenario: DICOM volumes are converted to HU16 grayscale slices
- **WHEN** preprocessing is run on a DICOM series
- **THEN** output images SHALL store HU values in a reversible uint16 encoding (e.g., offset + clip)
- **AND** windowing (including random windowing) SHALL be applied during training-time loading, not baked into Phase 2 outputs.

### Requirement: Visual Data Validation Artifacts
The system SHALL provide a simple, reproducible visual validation step for Phase 2 outputs.

#### Scenario: Ten random samples are generated for inspection
- **WHEN** Phase 2 validation is executed
- **THEN** the system SHALL generate 10 random sample images from the preprocessed dataset
- **AND** outputs SHALL be suitable for quick visual inspection (e.g., saved PNGs and/or a contact sheet) to confirm lung texture is visible and not collapsed to black/white blobs.

## ADDED Requirements

### Requirement: Deterministic Train/Validation Split
The system SHALL support reserving a deterministic validation split comprising 10% of the processed dataset prior to Phase 5 training.

#### Scenario: Split is reproducible
- **WHEN** a split is generated with a fixed seed and an immutable dataset manifest as input
- **THEN** the resulting train/validation membership SHALL be identical across reruns
- **AND** the split SHALL be written as a manifest file that can be referenced by training and monitoring tools.

### Requirement: No Cross-Split Leakage
The system SHALL prevent training from ingesting any sample assigned to the reserved validation split.

#### Scenario: Training excludes validation samples
- **WHEN** a training run is started with a split manifest configured
- **THEN** the training dataloader SHALL exclude all validation members
- **AND** the run configuration/logs SHALL record which split manifest was used.

### Requirement: Patient/Study-Level Isolation
The system SHALL support constructing the validation split at a patient/study granularity to avoid leakage from adjacent slices in the same volume.

#### Scenario: Patient volumes do not span splits
- **WHEN** the split is constructed from volumetric CT data with patient/study identifiers
- **THEN** all slices/images from a given patient/study SHALL be assigned to exactly one split (train OR validation).

### Requirement: Split Provenance Recorded
The split manifest SHALL record enough provenance to support reproducibility and auditing.

#### Scenario: Manifest captures provenance
- **WHEN** a split manifest is created
- **THEN** it SHALL record the source dataset manifest reference, the split rule (10%), and the seed
- **AND** it SHALL record creation time and counts for each split.
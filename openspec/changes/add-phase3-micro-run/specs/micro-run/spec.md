## ADDED Requirements

### Requirement: DINOv3 Micro-Run Training Loop
The system SHALL provide a minimal end-to-end DINOv3 training loop suitable for short "micro runs" on Strix Halo.

#### Scenario: Single-device micro-run executes
- **WHEN** a user runs the Phase 3 micro-run entrypoint on a Phase 1 + Phase 2 configured machine
- **THEN** the training loop SHALL complete a short run without runtime errors under ROCm
- **AND** the loop SHALL report step, loss, and throughput information to stdout (or a log file) for debugging.

### Requirement: Gram Anchoring Enabled
The system SHALL support enabling Gram Anchoring in the DINOv3 objective during micro-runs.

#### Scenario: Gram Anchoring is active
- **WHEN** the micro-run is started with Gram Anchoring enabled
- **THEN** the loss computation SHALL include the Gram Anchoring component
- **AND** the run configuration SHALL record that Gram Anchoring was enabled and with what coefficient(s).

### Requirement: Deterministic 1,000-Image Subset Mode
The system SHALL support a deterministic micro dataset selection of exactly 1,000 preprocessed images.

#### Scenario: Subset is reproducible
- **WHEN** a user runs micro-run with a fixed subset seed
- **THEN** the same 1,000 image identifiers/paths SHALL be selected across runs
- **AND** the subset definition SHALL be recorded alongside logs/checkpoints to support reproducibility.

### Requirement: Checkpoint Save and Resume
The system SHALL save and resume micro-run training from checkpoints at a configurable interval.

#### Scenario: Training resumes without losing state
- **WHEN** training is resumed from the latest checkpoint
- **THEN** model and optimizer state SHALL be restored
- **AND** the training step counter (and RNG state if applicable) SHALL be restored so loss continues smoothly.

### Requirement: Overfit Test Exit Criteria
The system SHALL define an "overfit" micro-run mode as a formal Phase 3 exit test.

#### Scenario: Loss drops near zero
- **WHEN** micro-run overfit mode is executed on the 1,000-image subset for approximately 1 hour
- **THEN** the reported loss SHALL decrease substantially and approach near-zero
- **AND** the run SHALL emit enough logging to confirm the gradient path is connected.

### Requirement: Restart Test Exit Criteria
The system SHALL define a restart/resume test as a formal Phase 3 exit test.

#### Scenario: Ctrl+C then resume continues
- **WHEN** a user interrupts micro-run with Ctrl+C and then restarts it with resume enabled
- **THEN** the run SHALL load the latest checkpoint
- **AND** the loss curve SHALL continue from approximately the previous value rather than restarting from scratch.

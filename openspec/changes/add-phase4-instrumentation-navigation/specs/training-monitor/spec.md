## ADDED Requirements

### Requirement: Offline Training Monitor Entry Point
The system SHALL provide a "training monitor" entry point that can be executed independently of the training loop to compute health metrics and visualizations for a specified checkpoint/run.

#### Scenario: Monitor runs safely alongside training
- **WHEN** the monitor is executed while a training job is running
- **THEN** the monitor SHALL operate in a read-only manner with respect to training state (e.g., checkpoints)
- **AND** the monitor SHALL write outputs to a separate directory without modifying training artifacts.

### Requirement: Fixed-Sample Attention Map Visualization
The training monitor SHALL generate an attention map visualization for a fixed, known test sample to enable day-over-day comparison.

#### Scenario: Attention visualization is generated
- **WHEN** the monitor is executed with a configured fixed test sample identifier
- **THEN** the monitor SHALL output an attention visualization artifact (e.g., a PNG)
- **AND** the output SHALL record the checkpoint identifier (step/epoch/hash) used to generate it.

### Requirement: Embedding Dispersion Metric
The training monitor SHALL compute an embedding-dispersion metric (standard deviation) over a fixed batch of samples as a feature-collapse indicator.

#### Scenario: Dispersion metric is reported
- **WHEN** the monitor is executed with a configured fixed batch definition
- **THEN** the monitor SHALL compute the standard deviation of embeddings (and any additional supporting statistics)
- **AND** the monitor SHALL emit the results as machine-readable output (e.g., JSON).

### Requirement: Reproducible Monitor Configuration
The training monitor SHALL be configurable in a reproducible way so that repeated executions are comparable across days.

#### Scenario: Monitor settings are persisted
- **WHEN** the monitor produces its outputs
- **THEN** it SHALL record the configuration used (sample ID, batch definition, seed, model/encoder settings)
- **AND** it SHALL record the exact checkpoint reference it evaluated.
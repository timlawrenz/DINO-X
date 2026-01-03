## ADDED Requirements

### Requirement: Virtual Inflation Dataset Wrapper
The system SHALL provide a "virtual inflation" mechanism that makes a small on-disk image set behave like a much larger dataset for DataLoader stress testing.

#### Scenario: Inflated index behaves like a large dataset
- **WHEN** the tuning tool is configured with an underlying dataset of N samples and an inflation factor K
- **THEN** the DataLoader-visible dataset length SHALL be N*K
- **AND** samples SHALL map deterministically to the underlying dataset indices (repeatable across runs).

### Requirement: Throughput Grid Search Runner
The system SHALL provide a throughput tuning entrypoint that iterates a grid of DataLoader and training-step parameters to locate the throughput knee and the OOM point.

The tuning entrypoint SHOULD support gradient accumulation (effective batch sizing) so that users can evaluate `effective_batch = microbatch * grad_accum_steps` while keeping the microbatch within memory limits.

#### Scenario: Grid search covers parameter combinations
- **WHEN** the tuning tool is run with configured sets of `batch_size`, `num_workers`, and `pin_memory`
- **THEN** the tool SHALL attempt each combination in a deterministic order
- **AND** it SHALL record success/failure for each attempted combination.

### Requirement: Safe Out-Of-Memory Handling
The tuning entrypoint SHALL detect out-of-memory failures and continue the grid search without terminating the entire run.

#### Scenario: OOM is recorded and the run continues
- **WHEN** a parameter combination triggers an out-of-memory condition
- **THEN** the tool SHALL record the failure (including the parameters)
- **AND** it SHALL continue to the next combination that may be viable.

### Requirement: Bottleneck Classification Metrics
The tuning entrypoint SHALL measure step-time components sufficient to classify whether the run is compute-bound or IO-bound.

#### Scenario: IO vs compute signal is emitted
- **WHEN** the tool executes benchmark steps for a given parameter combination
- **THEN** it SHALL report separate timing measurements for data decode/transform and for forward/backward/optimizer step
- **AND** it SHALL emit an aggregate throughput metric (images/sec) for the combination.

### Requirement: Machine-Readable Results Output
The tuning entrypoint SHALL emit results in a machine-readable format suitable for later analysis.

#### Scenario: Results are persisted
- **WHEN** the tool completes a grid search (or is interrupted after partial progress)
- **THEN** it SHALL write results to a CSV and/or JSON file
- **AND** it SHALL include the configuration grid and environment metadata in the output.

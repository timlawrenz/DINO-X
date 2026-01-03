## ADDED Requirements

### Requirement: Parameterized Model Configuration
The system SHALL support parameterized model architecture configuration to enable training different ViT sizes without code duplication.

#### Scenario: ViT-Large configuration is loaded
- **WHEN** a user runs the big-run script with `--config vit-large`
- **THEN** the system SHALL instantiate a ViT-Large model with patch size 14, dimension 1024, depth 24, and 16 attention heads
- **AND** the configuration SHALL be logged and saved in checkpoints for reproducibility.

#### Scenario: ViT-Giant configuration is loaded
- **WHEN** a user runs the big-run script with `--config vit-giant`
- **THEN** the system SHALL instantiate a ViT-Giant model with parameters exceeding 1B total parameters
- **AND** the configuration SHALL support memory requirements beyond 24GB VRAM.

#### Scenario: Custom configuration via CLI arguments
- **WHEN** a user provides explicit architecture parameters (`--vit-patch`, `--vit-dim`, `--vit-depth`, `--vit-heads`)
- **THEN** the system SHALL override preset configuration with the specified values
- **AND** the system SHALL validate that the resulting model fits within available hardware memory.

### Requirement: Hardware Target Selection
The system SHALL support both CUDA (NVIDIA) and ROCm (AMD) hardware targets with automatic detection and optimization.

#### Scenario: CUDA device detected and optimized
- **WHEN** the script runs on a system with NVIDIA GPUs
- **THEN** the system SHALL automatically detect CUDA runtime
- **AND** apply CUDA-specific optimizations (e.g., optimal num_workers, pin_memory settings from Phase 4.5).

#### Scenario: ROCm device detected and optimized
- **WHEN** the script runs on a system with AMD GPUs (Strix Halo)
- **THEN** the system SHALL automatically detect ROCm runtime
- **AND** apply ROCm-specific optimizations and utilize unified memory capabilities.

#### Scenario: Manual hardware target override
- **WHEN** a user specifies `--device cuda` or `--device rocm`
- **THEN** the system SHALL use the specified runtime regardless of auto-detection
- **AND** warn if the requested device is not available.

### Requirement: Phase 5a Validation Run (ViT-Large)
The system SHALL support a fixed-length validation run targeting 384 training steps on a 24GB GPU.

#### Scenario: ViT-Large validation run completes
- **WHEN** a user runs the script with `--config vit-large --max-steps 384 --num-workers 8`
- **THEN** training SHALL execute exactly 384 optimization steps (accounting for gradient accumulation)
- **AND** the run SHALL complete successfully on an RTX 4090 (24GB)
- **AND** checkpoints and monitoring outputs SHALL be saved for Phase 6a validation.

#### Scenario: Validation run proves convergence signals
- **WHEN** the ViT-Large validation run completes 384 steps
- **THEN** loss SHALL show decreasing trend
- **AND** monitoring SHALL confirm no feature collapse
- **AND** attention maps SHALL show meaningful structure.

### Requirement: Phase 5b Production Run (ViT-Giant)
The system SHALL support unlimited-length production training for ViT-Giant models exceeding 24GB memory requirements.

#### Scenario: ViT-Giant production run executes on amd395
- **WHEN** a user runs the script with `--config vit-giant` on AMD Strix Halo (amd395)
- **THEN** training SHALL utilize >24GB of unified memory
- **AND** training SHALL run continuously until manually stopped or convergence criteria are met
- **AND** the system SHALL demonstrate successful "memory wall" breaking.

#### Scenario: Multi-day training remains stable
- **WHEN** ViT-Giant production training runs for multiple days
- **THEN** the system SHALL maintain stable loss progression
- **AND** periodic monitoring SHALL detect and log any training anomalies
- **AND** automatic checkpoint rotation SHALL prevent disk space exhaustion.

### Requirement: Extended Checkpoint Management
The system SHALL provide robust checkpoint management for long-running training with automatic resumption.

#### Scenario: Periodic checkpoint saving
- **WHEN** training runs for extended periods
- **THEN** the system SHALL save checkpoints at configurable intervals (e.g., every 100 steps)
- **AND** each checkpoint SHALL include model state, optimizer state, RNG state, step counter, and configuration.

#### Scenario: Checkpoint rotation prevents disk exhaustion
- **WHEN** checkpoint count exceeds a configured maximum (e.g., keep last 5 checkpoints)
- **THEN** the system SHALL automatically delete oldest checkpoints
- **AND** always preserve the most recent checkpoint and optionally "milestone" checkpoints.

#### Scenario: Automatic resumption on restart
- **WHEN** training is restarted after interruption or shutdown
- **THEN** the system SHALL automatically detect and load the latest checkpoint
- **AND** resume training from the exact step count without manual intervention.

#### Scenario: Configuration mismatch detected
- **WHEN** a checkpoint is loaded with different model configuration than requested
- **THEN** the system SHALL halt with a clear error message
- **AND** suggest how to resolve the mismatch (e.g., use `--config` matching the checkpoint).

#### Scenario: Cross-hardware checkpoint portability
- **WHEN** a checkpoint saved on one hardware platform (e.g., 4090) is loaded on a different platform (e.g., amd395)
- **THEN** the system SHALL successfully load model and optimizer state regardless of hardware
- **AND** automatically apply the new hardware's optimization presets
- **AND** log a notice that hardware platform has changed.

### Requirement: Gradient Accumulation Support
The system SHALL support gradient accumulation to achieve large effective batch sizes on memory-constrained hardware.

#### Scenario: Effective batch size achieved via accumulation
- **WHEN** training runs with `--batch-size 64 --accumulation-steps 4`
- **THEN** the system SHALL accumulate gradients over 4 forward passes
- **AND** perform optimizer step with effective batch size of 256
- **AND** log both per-device batch size and effective batch size.

#### Scenario: Accumulation steps configurable per hardware
- **WHEN** different hardware targets require different accumulation strategies
- **THEN** configuration presets SHALL include appropriate accumulation_steps
- **AND** users MAY override via `--accumulation-steps` CLI argument.

### Requirement: Training Progress Monitoring
The system SHALL provide comprehensive progress monitoring including loss curves, throughput metrics, and health indicators.

#### Scenario: Real-time training metrics logged
- **WHEN** training is running
- **THEN** the system SHALL log step number, loss value, learning rate, and throughput (steps/sec) at regular intervals
- **AND** metrics SHALL be written to both stdout and structured log files.

#### Scenario: Memory monitoring prevents OOM
- **WHEN** VRAM usage approaches capacity
- **THEN** the system SHALL log a warning with current memory utilization
- **AND** suggest mitigation strategies (reduce batch size, enable gradient checkpointing).

#### Scenario: Phase 4 monitoring integration
- **WHEN** training reaches configured monitoring checkpoints (e.g., every 1000 steps)
- **THEN** the system SHALL invoke Phase 4 monitoring hooks
- **AND** generate attention map visualizations and embedding statistics
- **AND** save monitoring outputs alongside training logs.

### Requirement: Training Anomaly Detection
The system SHALL detect and warn about training anomalies that may indicate problems.

#### Scenario: Loss spike detected
- **WHEN** loss increases by more than 2x compared to recent moving average
- **THEN** the system SHALL log a warning with context (step number, loss history)
- **AND** continue training unless configured to halt on anomalies.

#### Scenario: Feature collapse detected
- **WHEN** embedding standard deviation falls below configured threshold
- **THEN** the system SHALL log a critical warning
- **AND** save an emergency checkpoint for post-mortem analysis.

#### Scenario: NaN or Inf loss detected
- **WHEN** loss becomes NaN or Inf
- **THEN** the system SHALL immediately halt training
- **AND** save an emergency checkpoint
- **AND** provide diagnostic information (last valid loss, learning rate, batch statistics).

### Requirement: Reproducibility and Provenance
The system SHALL ensure all training runs are reproducible and maintain complete provenance records.

#### Scenario: Configuration saved with checkpoint
- **WHEN** a checkpoint is saved
- **THEN** the complete configuration (CLI arguments, hardware info, random seeds) SHALL be saved
- **AND** the configuration SHALL be human-readable (e.g., JSON or YAML format).

#### Scenario: Git commit hash recorded
- **WHEN** training starts
- **THEN** the system SHALL record the current git commit hash of the repository
- **AND** warn if there are uncommitted changes
- **AND** save this information in checkpoints and logs.

#### Scenario: Data manifest hash recorded
- **WHEN** training starts
- **THEN** the system SHALL compute or load a hash of the training data manifest
- **AND** record it in logs and checkpoints to ensure data provenance.

### Requirement: Hardware-Specific Optimization Presets
The system SHALL apply hardware-specific optimizations automatically based on throughput tuning results from Phase 4.5.

#### Scenario: 4090 optimization preset applied
- **WHEN** training runs on RTX 4090
- **THEN** the system SHALL apply throughput-tuned parameters (num_workers, pin_memory, batch_size recommendations)
- **AND** log which optimization preset is active.

#### Scenario: amd395 optimization preset applied
- **WHEN** training runs on AMD Strix Halo (amd395)
- **THEN** the system SHALL apply unified-memory optimizations
- **AND** utilize throughput-tuned parameters specific to ROCm/unified memory architecture.

#### Scenario: Custom optimization override
- **WHEN** a user provides explicit optimization parameters
- **THEN** the system SHALL use provided values instead of presets
- **AND** log that custom optimizations are active.

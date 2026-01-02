## ADDED Requirements

### Requirement: Strix Halo Hardware Assembly and Thermals
The system SHALL define how the Strix Halo platform must be assembled and cooled to sustain multi-day DINO-X training runs.

#### Scenario: Server-class cooling configured
- **WHEN** the Strix Halo is installed in the target chassis
- **THEN** BIOS fan profiles SHALL be configured to a server/turbo-equivalent mode suitable for sustained 120W-class loads
- **AND** the configuration SHALL be documented so it can be reproduced on a second identical machine.

### Requirement: OS and Kernel Setup for gfx1151
The system SHALL require Linux with a kernel version that supports stable gfx1151 unified memory and ROCm 7.1 on Strix Halo.

#### Scenario: Kernel version validated
- **WHEN** Phase 1 platform bootstrap begins
- **THEN** the host SHALL be running Linux kernel 6.11 or newer (6.15+ preferred)
- **AND** the exact kernel version and configuration SHALL be recorded in project documentation.

### Requirement: ROCm 7.1 Stack Installation
The system SHALL install and validate ROCm 7.1 (with gfx1151 support) including core libraries required for ViT-Giant training.

#### Scenario: ROCm stack verified
- **WHEN** ROCm 7.1 is installed on Strix Halo
- **THEN** commands to query device support (e.g., `rocminfo`, `hipinfo`) SHALL list the Strix Halo GPU with gfx1151
- **AND** MIOpen and hipBLASLt libraries SHALL be available to PyTorch under the ROCm backend.

### Requirement: Flash Attention 2 Compilation on Strix Halo
The system SHALL compile and validate Flash Attention 2 with a Triton or Composable Kernel backend on the Strix Halo platform.

#### Scenario: Flash Attention kernel built
- **WHEN** the Phase 1 environment setup is run
- **THEN** Flash Attention 2 kernels SHALL compile successfully against ROCm 7.1 and the target kernel version
- **AND** a minimal test script SHALL confirm that a small attention forward pass executes without kernel errors.

### Requirement: Phase 1 Dot-Product Attention Validation
The system SHALL provide a minimal validation script that runs a 512×512 dot-product attention using the configured stack and confirms stability.

#### Scenario: 512×512 attention run succeeds
- **WHEN** the Phase 1 validation script is executed on the Strix Halo platform
- **THEN** it SHALL construct a 512×512 attention workload using PyTorch with ROCm and Flash Attention 2
- **AND** the script SHALL complete without crashes, GPU hangs, or numerical errors, recording basic timing and device information in its output.

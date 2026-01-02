# Change: Phase 1 Platform Bootstrap

## Why
Phase 1 of the roadmap requires proving that the Strix Halo platform can reliably run the core math stack (ROCm + Flash Attention) for DINO-X training before touching data or large-scale experiments.
This proposal formalizes the hardware and environment bring-up as a first-class capability with clear success criteria and reproducible steps.

## What Changes
- Define a `platform-bootstrap` capability that covers hardware assembly, thermal validation, OS/kernel configuration, ROCm 7.1 installation, and Flash Attention 2 compilation for Strix Halo.
- Add OpenSpec requirements and scenarios for Phase 1 covering hardware, OS/kernel, ROCm stack, and attention kernel validation.
- Document a minimal Python validation script that runs a 512×512 dot-product attention on the Strix Halo without crashes as the formal Phase 1 exit check.
- Track implementation tasks for wiring this capability into docs and scripts in the repository.

## Impact
- Affected specs: `specs/platform-bootstrap/spec.md` (new capability, added requirements).
- Affected code: hardware and environment docs under `docs/`, ROCm/Flash Attention setup scripts and configs under `src/` or `scripts/` (to be detailed during implementation).
- This change is non-breaking; it introduces an initial platform capability required by later roadmap phases.

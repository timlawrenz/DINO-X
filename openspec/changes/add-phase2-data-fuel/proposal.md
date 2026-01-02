# Change: Phase 2 Data (Fuel)

## Why
Phase 2 of the roadmap requires turning raw CT scans into a deterministic, model-ready dataset.
Because the project must remain safe to clone and collaborate on without risking accidental commits of protected raw medical data, raw inputs must live outside the repository.

## What Changes
- Define a `data-fuel` capability covering dataset acquisition, secure storage layout, preprocessing into **16-bit HU PNG slices**, and visual validation.
- Standardize a NAS-backed raw data location mounted at `/mnt/nas-ai-models/` while keeping the repository self-contained via **local symlinks**.
- Specify a directory layout and environment variables so all scripts can run on any machine with either:
  - the NAS mount available, or
  - an alternate data root configured explicitly.
- Require a lightweight validation artifact (10 random samples + contact sheet) to visually verify windowing and slice stacking quality.

## Impact
- Affected specs: `specs/data-fuel/spec.md` (new capability, added requirements).
- Affected code (future implementation): preprocessing scripts under `scripts/`, dataset documentation under `docs/`, and repository safety guards (e.g., `.gitignore`, symlink setup helper).
- Security/privacy: introduces a documented pattern to keep raw data off Git while enabling reproducible local paths.
- This change is non-breaking; it introduces Phase 2 capabilities required by later training phases.

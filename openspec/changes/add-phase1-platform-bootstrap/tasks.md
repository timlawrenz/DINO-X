## 1. Implementation
- [ ] 1.1 Document Strix Halo hardware assembly and thermal configuration (BIOS fan profile, chassis notes) aligned with Phase 1.
- [ ] 1.2 Document OS and kernel requirements (Linux 6.11+; preferred 6.15+) and produce a reproducible install/config guide.
- [ ] 1.3 Install and document ROCm 7.1 with gfx1151 support, including validation commands to confirm drivers and libraries (MIOpen, hipBLASLt, RCCL) are active.
- [ ] 1.4 Compile Flash Attention 2 using the Triton/CK backend on Strix Halo and record build flags, versions, and smoke tests.
- [ ] 1.5 Implement a minimal Python script that runs a 512×512 dot-product attention on the Strix Halo using the installed stack and exits cleanly.
- [ ] 1.6 Integrate the validation script into the repository (e.g., under `src/` or `scripts/`) and add a short how-to in `docs/hardware_setup.md` or a new Phase 1 doc.
- [ ] 1.7 (Optional) Add a simple CI or manual checklist item to run the Phase 1 validation script after environment changes.

# DINO-X Roadmap

## Phase 1: The Platform (Hardware & Environment)

Before looking at data, we must prove the Strix Halo can actually run the math.

- [x] **Hardware Assembly & Thermal Check**: Install the Strix Halo in the Framework/Desktop chassis. Configure fans to "Server/Turbo" mode in BIOS.
- [x] **OS & Kernel Setup**: Install Linux (Kernel 6.11+ required for Strix Halo NPU/GPU addressing).
- [x] **ROCm 7.1 Installation**: Install the gfx1151 specific drivers and libraries (miopen-hip, rccl).
- [x] **Flash Attention Compilation**: Compile Flash Attention 2 using the Triton backend.

**Implementation links (OpenSpec + docs):**
- OpenSpec change: `openspec/changes/add-phase1-platform-bootstrap/proposal.md`
- Tasks checklist: `openspec/changes/add-phase1-platform-bootstrap/tasks.md`
- Setup guide: `docs/hardware_setup.md`
- Validation script: `scripts/phase1_validate_attention.py`

**Success Criteria**: Run `scripts/phase1_validate_attention.py` to perform Dot-Product Attention on a 512x512 matrix and confirm it doesn't crash (prints `ok=true`).

## Phase 2: The Data (Fuel)

Raw medical data is unusable for Vision Transformers. We must transform it.

- [x] **Acquire LIDC-IDRI**: Download the dataset (approx. 120GB) from TCIA.
- [x] **Develop Preprocessing Pipeline**: Write the script to convert 16-bit DICOM volumes into "2.5D" RGB slices (using the windowing strategy: Lung/Soft-Tissue/Bone).
- [x] **Data Validation**: Generate a set of 10 random sample images and inspect them visually.
- [x] **Write Provenance Manifest**: Record counts/bytes and timestamp for raw + processed data.

**Implementation links (OpenSpec + docs):**
- OpenSpec change: `openspec/changes/add-phase2-data-fuel/proposal.md`
- Tasks checklist: `openspec/changes/add-phase2-data-fuel/tasks.md`
- Data guide: `docs/data_fuel.md`
- LIDC-IDRI acquisition: `docs/lidc_idri.md`
- Symlink bootstrap: `scripts/phase2_setup_data_root.sh`
- Preprocessing script: `scripts/phase2_preprocess_lidc_idri.py`
- Validation script: `scripts/phase2_validate_samples.py`
- Dataset manifest helper: `scripts/phase2_write_dataset_manifest.py`

**Success Criteria**: The PNGs should clearly show lung texture (not just black/white blobs) and represent 3 depth slices per image; a manifest JSON is generated under `data/processed/_manifests/`.

## Phase 3: The "Micro-Run" (Fail Fast)

Do not start the main training yet. We need to prove the code works on a tiny scale.

- [ ] **Implement DINOv3 Loop**: Port the training loop with Gram Anchoring enabled.
- [ ] **Implement Checkpointing**: Write the code to save/resume model state (.pth) every N steps.
- [ ] **The "Overfit" Test**:
  - Take only 1,000 images.
  - Train for 1 hour.
  - **Success Criteria**: Loss must drop significantly (near zero). This proves the model can learn and the gradient path is connected.
- [ ] **The "Restart" Test**: Interrupt the training (Ctrl+C), load the last checkpoint, and resume. Verify loss continues from where it left off.

## Phase 4: The Instrumentation (Navigation)

Flying blind for 15 days is dangerous. We need instruments.

- [ ] **Build the Monitor**: Write a script that runs safely alongside training (e.g., every 24h) to:
  - Generate an Attention Map visualization of a fixed test image.
  - Calculate the Standard Deviation of embeddings (to detect collapse).
- [ ] **Define Validation Set**: Isolate 10% of the data now so it never leaks into the training set.

## Phase 5: The "Big Run" (Execution)

The 15-day marathon.

- [ ] **Launch Training**: Run the script on the full dataset with Batch Size ~128+.
- [ ] **Daily Health Check**: Briefly check thermal logs and the "Monitor" outputs every morning.

## Phase 6: Validation (The Proof)

Did it actually work?

- [ ] **Attention Map Visualization**: Generate heatmaps for 50 random nodules. Do they light up?
- [ ] **Linear Probe Benchmark**: Freeze the backbone, train a simple classifier layer (Logistic Regression) on the "Malignancy" labels.
  - **Success Criteria**: AUC > 0.90.
- [ ] **Release**: Document weights and upload to Hugging Face.

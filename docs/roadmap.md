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
- [x] **Develop Preprocessing Pipeline**: Write the script to convert DICOM volumes into **16-bit lossless HU PNG slices** (one file per slice), deferring windowing to training for **random windowing** augmentation.
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

**Success Criteria**: Validation previews should clearly show lung texture (not just black/white blobs); the index CSV is generated and Phase 3 can assemble 3-slice inputs (z-1,z,z+1) from the stored HU16 slices; a manifest JSON is generated under `data/processed/_manifests/`.

## Phase 3: The "Micro-Run" (Fail Fast)

Do not start the main training yet. We need to prove the code works on a tiny scale.

- [x] **Implement DINOv3 Loop**: Port the training loop with Gram Anchoring enabled.
- [x] **Implement Checkpointing**: Write the code to save/resume model state (.pth) every N steps.
- [x] **The "Overfit" Test**:
  - Take only 1,000 images.
  - Train for 1 hour.
  - **Success Criteria**: Loss must drop significantly (near zero). This proves the model can learn and the gradient path is connected.
- [x] **The "Restart" Test**: Interrupt the training (Ctrl+C), load the last checkpoint, and resume. Verify loss continues from where it left off.

**Implementation links (OpenSpec + docs):**
- OpenSpec change: `openspec/changes/add-phase3-micro-run/proposal.md`
- Tasks checklist: `openspec/changes/add-phase3-micro-run/tasks.md`
- Micro-run guide: `docs/phase3_micro_run.md`
- Training script: `scripts/phase3_micro_run.py`

## Phase 4: The Instrumentation (Navigation)

Flying blind for 15 days is dangerous. We need instruments.

- [x] **Build the Monitor**: Write a script that runs safely alongside training (e.g., every 24h) to:
  - Generate an Attention Map visualization of a fixed test image.
  - Calculate the Standard Deviation of embeddings (to detect collapse).
- [x] **Define Validation Set**: Isolate 10% of the data now so it never leaks into the training set.

**Implementation links (OpenSpec + docs):**
- OpenSpec change: `openspec/changes/add-phase4-instrumentation-navigation/proposal.md`
- Tasks checklist: `openspec/changes/add-phase4-instrumentation-navigation/tasks.md`
- Phase 4 guide: `docs/phase4_instrumentation.md`
- Split manifest generator: `scripts/phase4_make_split_manifest.py`
- Training monitor: `scripts/phase4_monitor.py`

## Phase 4.5: Hyperparameter Grid Search (Throughput Tuning)

Strix Halo has massive unified memory capacity but comparatively lower memory bandwidth, so we need to experimentally find the throughput "knee of the curve" before starting the Big Run.

**Strategy: "Virtual Inflation"**
We don't need 120GB of unique files to test throughput; we need to force the DataLoader and GPU to work as if there were 120GB.

- [x] **Build Throughput Tuner**: Add `scripts/tune_throughput.py` that wraps the existing `PngDataset` in a "virtual epoch" loop.
- [x] **Inflate the Index**: Repeat a small local list of images many times in-memory so the DataLoader sees a very large dataset (e.g., 1,000 images → 1,000,000 samples).
- [x] **Grid Search Critical Parameters**:
  - `batch_size`: `[32, 64, 128, 192, 256, 512]` (find OOM threshold)
  - `num_workers`: `[0, 4, 8, 16, 24, 32]` (find CPU/RAM/unified-bandwidth choke)
  - `pin_memory`: `[True, False]` (APU behavior; sometimes `False` wins due to zero-copy)
- [x] **Profile the Bottleneck**: Measure data decode/transform time vs. forward/backward time to classify IO-bound vs compute-bound regimes.

**Implementation links (OpenSpec):**
- OpenSpec change: `openspec/changes/add-throughput-tuning/proposal.md`
- Tasks checklist: `openspec/changes/add-throughput-tuning/tasks.md`

**Current tuned decision (Strix Halo)**
- **Target architecture (Phase 5 Large-first):** ViT-Large (`--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16`)
- **Target effective batch:** `256+` (DINO stability)
- **Tuned config:** `--batch-sizes 128 --grad-accum-steps 2 --grad-checkpoint`
- **Observed peak throughput:** ~23.5 img/s (Strix Halo, Triton enabled)

**Success Criteria**: Produce a Phase 5 launch configuration that (1) reaches the maximum stable effective batch size (≥256) without OOM/kernel instability, and (2) selects `num_workers`/`pin_memory` at the throughput knee so the GPU stays busy without saturating unified memory bandwidth.

## Phase 5: The "Big Run" (Execution)

We execute this phase **twice** to validate the process before attempting the memory-breaking Giant run.

### Phase 5a: ViT-Large Validation Run (4090)

A focused validation run to verify end-to-end training convergence within budget constraints.

- [x] **Launch Training**: Run ViT-Large on RTX 4090 (24GB) with reduced training steps.
  - **Configuration**: `--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16 --num-workers 8`
  - **Training Steps**: 384 steps (192 × 2 accumulation steps)
  - **Status**: Launched successfully. Fixed critical collapse issue by implementing multi-view augmentation (2 views) and teacher centering (`DINOLoss`).
  - **Goal**: Verify convergence signals and validate the training process
- [ ] **Daily Health Check**: Monitor thermal logs and validation outputs.

### Phase 5b: ViT-Giant Production Run (amd395)

The full-scale marathon after validation success.

- [ ] **Launch Training**: Run ViT-Giant on AMD Strix Halo (amd395, 128GB Unified Memory).
  - **Configuration**: Memory-breaking ViT-Giant configuration (utilizing >24GB VRAM)
  - **Training Duration**: 15-day marathon with full dataset
  - **Goal**: Break the 24GB memory wall and train the production foundation model
- [ ] **Daily Health Check**: Monitor thermal logs and validation outputs throughout the 15-day run.

## Phase 6: Validation (The Proof)

We execute validation **twice** - once after each Big Run.

### Phase 6a: ViT-Large Validation & Release

Validate the ViT-Large model and release it as open source.

- [ ] **Attention Map Visualization**: Generate heatmaps for 50 random nodules. Do they light up?
- [ ] **Linear Probe Benchmark**: Freeze the backbone, train a simple classifier layer (Logistic Regression) on the "Malignancy" labels.
  - **Success Criteria**: AUC > 0.90.
- [ ] **Release**: Document ViT-Large weights and upload to Hugging Face as an open-source validation model.

### Phase 6b: ViT-Giant Validation & Release

Validate the production ViT-Giant model and release it as the flagship foundation model.

- [ ] **Attention Map Visualization**: Generate heatmaps for 50 random nodules. Do they light up?
- [ ] **Linear Probe Benchmark**: Freeze the backbone, train a simple classifier layer (Logistic Regression) on the "Malignancy" labels.
  - **Success Criteria**: AUC > 0.90.
- [ ] **Release**: Document ViT-Giant weights and upload to Hugging Face as the production foundation model.

## 1. Script Infrastructure
- [x] 1.1 Create `scripts/phase5_big_run.py` as the main training entrypoint
- [x] 1.2 Implement model configuration presets (ViT-Large and ViT-Giant)
- [x] 1.3 Add CLI argument parser with configuration selection (`--config vit-large` or `--config vit-giant`)
- [x] 1.4 Add hardware target detection/override (`--device cuda` or `--device rocm`)

## 2. Model Configuration System
- [x] 2.1 Define ViT-Large preset: `--vit-patch 14 --vit-dim 1024 --vit-depth 24 --vit-heads 16`
- [x] 2.2 Define ViT-Giant preset: patch/dim/depth/heads for >1B parameters
- [x] 2.3 Implement configuration validation (ensure parameters are valid for target hardware)
- [x] 2.4 Add configuration logging (print/save active config at training start)

## 3. Training Loop
- [x] 3.1 Integrate DINOv3 training loop with Gram Anchoring from Phase 3
- [x] 3.2 Add gradient accumulation support for large effective batch sizes
- [x] 3.3 Implement training step limit for Phase 5a (384 steps) and unlimited for Phase 5b
- [x] 3.4 Add throughput logging (steps/sec, samples/sec)

## 4. Checkpoint Management
- [x] 4.1 Extend Phase 3 checkpoint system for long-running training
- [x] 4.2 Add periodic checkpoint saving (every N steps, configurable)
- [x] 4.3 Implement checkpoint rotation (keep last K checkpoints to save disk space)
- [x] 4.4 Add automatic resume from latest checkpoint on script restart
- [x] 4.5 Save training configuration in checkpoint for reproducibility

## 5. Hardware Optimization
- [x] 5.1 Detect CUDA vs ROCm runtime and apply appropriate optimizations
- [x] 5.2 Apply throughput-tuned parameters from Phase 4.5 for each hardware target
- [x] 5.3 Add memory monitoring (log VRAM usage, warn on approaching OOM)
- [x] 5.4 Implement hardware-specific batch size recommendations

## 6. Monitoring Integration
- [x] 6.1 Integrate Phase 4 monitoring hooks (attention maps, embedding std dev)
- [x] 6.2 Add training progress logging (loss curves, learning rate schedule)
- [x] 6.3 Generate periodic health reports (thermal, memory, throughput)
- [x] 6.4 Implement early warning system for training anomalies (loss spikes, collapse detection)

## 7. Documentation
- [x] 7.1 Create `docs/phase5_big_run.md` with usage guide
- [x] 7.2 Document Phase 5a execution (ViT-Large validation run)
- [x] 7.3 Document Phase 5b execution (ViT-Giant production run)
- [x] 7.4 Add example commands and expected outputs
- [x] 7.5 Document checkpoint management and resumption procedures

## 8. Testing
- [x] 8.1 Test ViT-Large configuration on 4090 (dry run, 10 steps)
- [x] 8.2 Test ViT-Giant configuration on amd395 (dry run, 10 steps)
- [x] 8.3 Verify checkpoint save/resume cycle for both configurations
- [x] 8.4 Validate configuration presets produce expected model sizes
- [x] 8.5 Test hardware detection and optimization path selection

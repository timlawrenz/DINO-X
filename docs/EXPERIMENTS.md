# Experiments Log

Updated: 2026-01-09 16:35:51

| Run ID | Model | Eff Batch | LR | Warmup | T-Temp | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20260109_162920_4090_512px_DeepFreeze` | vit-large (p14) | 256 | 2e-05 | 3000 | 0.02 | Running |  |
| `20260109_104007_4090_LowLR_IceAge` | vit-large (p14) | 256 | 5e-05 | 1000 | 0.02 | Completed | Success. Best stability. Found Golden Zone LR ~2e-5. Drifted up at 5e-5. |
| `20260108_203723_4090_IceAge` | vit-large (p14) | 256 | 0.0002 | 1000 | 0.02 | Completed | Success! Broken entropy wall (6.78). Frozen Teacher (0.9995) + Sharp Temp (0.02). |
| `20260108_175149_4090_dim2048_test` | vit-large (p14) | 256 | 0.0002 | 1000 | 0.04 | Completed |  |
| `20260108_084549_4090_224px_slower` | vit-large (p14) | 256 | 0.0002 | 5000 | 0.05 | Completed |  |
| `20260108_003002_amd395_giant_prod_run` | vit-large (p14) | 256 | 0.0002 | 5000 | 0.04 | Completed |  |
| `20260107_211157_4090_224px` | vit-large (p14) | 256 | 0.0005 | 2500 | 0.05 | Stopped | Failed. Flatlined at 9.01. LR 5e-4 too high. |
| `20260107_181724_4090_warm_up_test` | vit-large (p16) | 256 | 0.0001 | 200 | 0.03 | Completed | Success/Stopped. Proof of throughput. Hit 9.01 wall. Backbone good. |
| `20260107_181716_4090_warm_up_test` | vit-large (p16) | 256 | 0.0001 | 2500 | 0.03 | Stopped |  |
| `20260106_223221_4090_large_teacher0-015` | vit-large (p16) | 256 | 0.0005 |  | 0.015 | Completed |  |
| `20260104_194515_4090_large_teacher0-03` | vit-large (p16) | 256 | 0.0001 |  | 0.03 | Stopped |  |
| `20260104_181120_amd395_giant_64x4` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Completed |  |
| `20260104_170333_amd395_giant_128x2` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Stopped |  |
| `20260104_165932_amd395_giant_128x2` | vit-large (p14) | 256 | 0.0005 |  | 0.02 | Stopped |  |
| `20260104_145936` | vit-large (p14) | 256 | 0.0005 |  | 0.04 | Stopped |  |
| `20260104_123726_4090_large_teacher0-025` | vit-large (p16) | 256 | 0.0005 |  | 0.025 | Completed |  |
| `20260104_110654_4090_large_teacher0-02` | vit-large (p16) | 256 | 0.0005 |  | 0.02 | Completed |  |
| `20260104_093402` | vit-large (p16) | 256 | 0.0005 |  | 0.04 | Completed |  |
| `20260103_175300` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_165249` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_163425` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_162833` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_155511` | vit-large (p16) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_155138` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_155103` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_155028` | vit-large (p14) | 288 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154819` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154759` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_154642` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_152502` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_150009` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_144457` | vit-large (p14) | 16 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_144115` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143710` | vit-large (p14) | 256 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143550` | vit-large (p14) | 192 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_143247` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_141953` | vit-large (p14) | 32 | 0.0001 |  | 0.04 | Completed |  |
| `20260103_141940` | vit-large (p14) | 32 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_141803` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Stopped |  |
| `20260103_141734` | vit-large (p14) | 64 | 0.0001 |  | 0.04 | Stopped |  |

# DINOv3 Architecture & Integration Guide

**Last Updated**: 2026-02-20  
**Purpose**: Reference documentation for integrating DINOv3 spatial features into vision models.

---

## Overview

DINOv3 is Meta's third-generation self-supervised Vision Transformer, trained on 1.7B images. It produces high-quality dense features suitable for spatial conditioning in generative models.

**Key Characteristics**:
- Self-supervised training (no labels required)
- Strong spatial understanding
- Variable-length sequence support via RoPE
- Produces both global (CLS) and local (patch) features

**Models Available**:
- ViT-S/16, ViT-B/16, ViT-L/16, ViT-H/16, ViT-7B/16 (web images)
- ConvNeXt variants (web images)
- ViT-L/16, ViT-7B/16 (satellite imagery)

---

## Architecture Deep Dive

### 1. Positional Encoding: RoPE (Rotary Position Embeddings)

**Critical Discovery**: DINOv3 uses **RoPE**, not learned/interpolated absolute positional embeddings.

**What is RoPE?**
- Rotary position encoding that naturally supports any resolution
- No interpolation needed for different aspect ratios
- Computes positional embeddings on-the-fly based on input dimensions
- Axial encoding (H and W dimensions handled separately)

**Implications**:
- ✅ Variable-length sequences are natively supported
- ✅ No fixed grid or interpolation artifacts
- ✅ Patches correspond precisely to spatial locations
- ✅ Can process any input size (H×W must be multiples of patch_size)

**Source**: `dinov3/layers/rope_position_encoding.py`

```python
# RoPE forward pass (simplified)
def forward(self, H: int, W: int):
    # Create coordinate grid
    coords_h = torch.arange(0.5, H) / H  # [H]
    coords_w = torch.arange(0.5, W) / W  # [W]
    coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # [H, W, 2]
    
    # Compute angles and sin/cos
    angles = 2 * pi * coords / self.periods  # [HW, D]
    return torch.sin(angles), torch.cos(angles)
```

### 2. Token Structure

DINOv3 outputs a structured sequence of tokens:

```
[CLS] [patch_1] [patch_2] ... [patch_N] [register_1] [register_2] [register_3] [register_4]
  ^         ^--- variable length ---^              ^---- 4 registers ----^
  |                                  
  Global token for classification/retrieval
```

**Token Counts** (for ViT models):
- `total_tokens = 1 (CLS) + num_patches + 4 (registers)`
- `num_patches = (H ÷ patch_size) × (W ÷ patch_size)`

**Register Tokens** (new in DINOv3):
- 4 learnable "memory slots" for global information
- Reduce high-norm artifacts in patch tokens
- Improve attention map quality for dense tasks
- **Not used for spatial features** (exclude when extracting patches)

### 3. Patch Grid Calculation

**Formula**:
```python
# Input dimensions (must be multiples of patch_size=16)
H = round(target_height / 16) * 16
W = round(target_width / 16) * 16

# Patch grid dimensions
patch_grid_h = H // 16
patch_grid_w = W // 16

# Total patches
num_patches = patch_grid_h * patch_grid_w
```

**Common Resolutions** (for 16×16 patches):

| Image Size | Patch Grid | Patch Count | Total Tokens |
|------------|------------|-------------|--------------|
| 1024×1024  | 64×64      | 4096        | 4101         |
| 832×1216   | 52×76      | 3952        | 3957         |
| 1216×832   | 76×52      | 3952        | 3957         |
| 768×1280   | 48×80      | 3840        | 3845         |
| 704×1344   | 44×84      | 3696        | 3701         |

### 4. Spatial Correspondence

**Key Insight**: Each patch corresponds to a specific spatial region in the image.

```
Image (832×1216)
├─> Resize to 832×1216 (no center-crop!)
├─> Patchify into 52×76 grid
└─> Each patch(i,j) represents region(i,j) in original image

Patch(0,0) = top-left corner
Patch(51,75) = bottom-right corner
```

**Spatial Alignment Requirements**:
1. **No center-crop** during preprocessing (preserves full image)
2. **Aspect-correct resize** to bucket dimensions
3. **Round to multiples of 16** (not 14!)
4. Patches maintain 1:1 correspondence with image regions

---

## Common Pitfalls & Solutions

### ❌ Pitfall 1: Rounding to Wrong Multiple

**Problem**:
```python
# WRONG: Rounding to multiples of 14
dino_w = round(target_width / 14) * 14  # ❌
dino_h = round(target_height / 14) * 14
```

**Why it's wrong**:
- DINOv3 patch_size is **16**, not 14
- Creates misaligned inputs (e.g., 826×1218 instead of 832×1216)
- Breaks spatial correspondence between patches and image regions

**Solution**:
```python
# CORRECT: Round to multiples of 16
dino_w = round(target_width / 16) * 16  # ✅
dino_h = round(target_height / 16) * 16
```

### ❌ Pitfall 2: Using Center-Crop

**Problem**:
```python
# WRONG: Center-crop destroys spatial alignment
processor(image, size={"height": 1216, "width": 832}, do_center_crop=True)
```

**Why it's wrong**:
- Crops out edges of the image
- Patch(0,0) no longer corresponds to top-left corner
- Spatial features don't align with DiT latent positions

**Solution**:
```python
# CORRECT: Disable center-crop, use full image
processor(image, size={"height": 1216, "width": 832}, do_center_crop=False)
```

### ❌ Pitfall 3: Assuming Fixed Patch Count

**Problem**:
```python
# WRONG: Assuming fixed 3880 patches
expected_patches = 3880  # ❌
```

**Why it's wrong**:
- DINOv3 patch count varies with input size
- Different buckets produce different patch counts
- RoPE supports variable-length sequences natively

**Solution**:
```python
# CORRECT: Calculate based on input dimensions
num_patches = (H // 16) * (W // 16)  # ✅
```

### ❌ Pitfall 4: Including Register Tokens in Patches

**Problem**:
```python
# WRONG: Extracting all non-CLS tokens
patches = outputs.last_hidden_state[0, 1:, :]  # Includes registers! ❌
```

**Why it's wrong**:
- Last 4 tokens are register tokens (global info)
- Not spatially localized
- Pollutes spatial feature extraction

**Solution**:
```python
# CORRECT: Exclude CLS (0) and registers (last 4)
total_tokens = outputs.last_hidden_state.shape[1]
num_patches = total_tokens - 5  # -1 CLS, -4 registers
patches = outputs.last_hidden_state[0, 1:num_patches+1, :]  # ✅
```

---

## Integration Best Practices

### 1. Preprocessing Pipeline

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load model and processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")

# Load image
image = Image.open("image.jpg")

# Determine target dimensions (from your bucketing system)
target_width, target_height = 832, 1216  # Example

# Round to multiples of 16
dino_w = round(target_width / 16) * 16
dino_h = round(target_height / 16) * 16

# Preprocess: resize without center-crop
inputs = processor(
    images=image,
    size={"height": dino_h, "width": dino_w},
    do_center_crop=False,  # CRITICAL!
    do_resize=True,
    return_tensors="pt"
)

# Run model
with torch.no_grad():
    outputs = model(**inputs)

# Extract features
cls_token = outputs.last_hidden_state[0, 0, :]  # [1024]

total_tokens = outputs.last_hidden_state.shape[1]
num_patches = total_tokens - 5
patches = outputs.last_hidden_state[0, 1:num_patches+1, :]  # [num_patches, 1024]

print(f"CLS shape: {cls_token.shape}")
print(f"Patches shape: {patches.shape}")
print(f"Expected patches: {(dino_h // 16) * (dino_w // 16)}")
```

### 2. Cross-Attention Integration

**Architecture**: Concatenated sequence for cross-attention.

```python
# In DiT model
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.text_proj = nn.Linear(4096, hidden_size)  # T5 embeddings
        self.dino_proj = nn.Linear(1024, hidden_size)  # DINO CLS
        self.dino_patch_proj = nn.Linear(1024, hidden_size)  # DINO patches
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads)
        
    def forward(self, x, t5_emb, dino_cls, dino_patches, context_mask):
        """
        Args:
            x: [B, N, hidden_size] - DiT latent tokens
            t5_emb: [B, 500, 4096] - T5 text embeddings
            dino_cls: [B, 1, 1024] - DINO CLS token
            dino_patches: [B, num_patches, 1024] - DINO spatial patches (variable length!)
            context_mask: [B, 500 + 1 + num_patches] - attention mask
        """
        # Project to common dimension
        t5_proj = self.text_proj(t5_emb)  # [B, 500, hidden_size]
        cls_proj = self.dino_proj(dino_cls)  # [B, 1, hidden_size]
        patches_proj = self.dino_patch_proj(dino_patches)  # [B, num_patches, hidden_size]
        
        # Concatenate context: [text, global_CLS, spatial_patches]
        context = torch.cat([t5_proj, cls_proj, patches_proj], dim=1)
        # Shape: [B, 500 + 1 + num_patches, hidden_size]
        
        # Cross-attention
        out, _ = self.cross_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=~context_mask
        )
        
        return out
```

### 3. Classifier-Free Guidance (CFG)

**Strategy**: Drop all DINO features together (CLS + patches).

```python
def apply_cfg_dropout(dino_cls, dino_patches, cfg_prob=0.1):
    """
    Drop DINO features for CFG with probability cfg_prob.
    
    Args:
        dino_cls: [B, 1, 1024]
        dino_patches: [B, num_patches, 1024]
        cfg_prob: Dropout probability
    
    Returns:
        (dino_cls, dino_patches) with some samples replaced by null embeddings
    """
    B = dino_cls.shape[0]
    
    # Generate dropout mask
    keep_mask = torch.rand(B, device=dino_cls.device) > cfg_prob
    keep_mask = keep_mask.view(B, 1, 1)
    
    # Null embeddings (learned parameters)
    null_cls = nn.Parameter(torch.zeros(1, 1, 1024))  # Global null
    null_patch = nn.Parameter(torch.zeros(1, 1, 1024))  # Spatial null
    
    # Apply dropout
    dino_cls_out = torch.where(keep_mask, dino_cls, null_cls)
    
    # Broadcast null_patch to match num_patches
    num_patches = dino_patches.shape[1]
    null_patches = null_patch.expand(1, num_patches, 1024)
    keep_mask_patches = keep_mask.expand(-1, num_patches, 1024)
    dino_patches_out = torch.where(keep_mask_patches, dino_patches, null_patches)
    
    return dino_cls_out, dino_patches_out
```

### 4. Validation & Debugging

**Sanity checks** to verify correct integration:

```python
def validate_dinov3_extraction(image_path, target_w, target_h):
    """Validate DINOv3 extraction produces correct patch count."""
    from PIL import Image
    
    # Load and preprocess
    image = Image.open(image_path)
    dino_w = round(target_w / 16) * 16
    dino_h = round(target_h / 16) * 16
    
    inputs = processor(
        images=image,
        size={"height": dino_h, "width": dino_w},
        do_center_crop=False,
        return_tensors="pt"
    )
    
    # Extract
    outputs = model(**inputs)
    total_tokens = outputs.last_hidden_state.shape[1]
    num_patches = total_tokens - 5
    
    # Expected
    expected_patches = (dino_h // 16) * (dino_w // 16)
    
    # Validate
    assert num_patches == expected_patches, \
        f"Patch count mismatch! Got {num_patches}, expected {expected_patches}"
    
    print(f"✓ Validation passed: {num_patches} patches for {dino_w}×{dino_h}")
    return True
```

---

## Technical Specifications

### Model Configurations

| Model | Params | Hidden Size | Heads | Layers | Patch Size | Registers |
|-------|--------|-------------|-------|--------|------------|-----------|
| ViT-S/16 | 21M | 384 | 6 | 12 | 16 | 4 |
| ViT-B/16 | 86M | 768 | 12 | 12 | 16 | 4 |
| ViT-L/16 | 300M | 1024 | 16 | 24 | 16 | 4 |
| ViT-H/16 | 840M | 1280 | 16 | 32 | 16 | 4 |
| ViT-7B/16 | 6716M | 4096 | 64 | 48 | 16 | 4 |

### Output Dimensions

**CLS Token**:
- Shape: `[hidden_size]`
- ViT-L/16: `[1024]`
- ViT-7B/16: `[4096]`

**Patch Tokens**:
- Shape: `[num_patches, hidden_size]`
- num_patches: Variable, depends on input size
- ViT-L/16: `[num_patches, 1024]`

**Total Sequence**:
- Shape: `[1 + num_patches + 4, hidden_size]`
- Example (832×1216, ViT-L/16): `[3957, 1024]`

### Memory Requirements (ViT-L/16)

**Inference** (batch_size=1):
- Model weights: ~1.2 GB (fp32) or ~600 MB (fp16)
- Activation memory (1024×1024 input): ~500 MB
- Total: ~1.7 GB GPU memory

**Training considerations**:
- Gradient checkpointing recommended for large models
- Mixed precision (fp16/bf16) saves memory
- Patches can be precomputed and cached

---

## Performance Characteristics

### Inference Speed (ViT-L/16, A100 GPU)

| Input Size | Patches | Time (ms) | Throughput |
|------------|---------|-----------|------------|
| 224×224    | 196     | ~15 ms    | ~67 img/s  |
| 512×512    | 1024    | ~35 ms    | ~29 img/s  |
| 1024×1024  | 4096    | ~120 ms   | ~8 img/s   |

**Notes**:
- Times include preprocessing + forward pass
- Batch inference can improve throughput
- Smaller models (ViT-B/16) are ~2x faster

### Quality vs. Size Trade-offs

**For Spatial Conditioning**:
- ViT-S/16: Fast, good for prototyping
- **ViT-L/16**: Best balance (recommended)
- ViT-7B/16: Highest quality, but slow and memory-intensive

**Recommendation**: Start with ViT-L/16, only upgrade to 7B if quality justifies cost.

---

## Comparison to Alternatives

### DINOv3 vs. CLIP

| Feature | DINOv3 | CLIP |
|---------|--------|------|
| Training | Self-supervised | Contrastive (text+image) |
| Spatial Features | Excellent | Good |
| Text Alignment | None (separate text model) | Native |
| Patch Granularity | 16×16 | 14×14 or 16×16 |
| Best For | Dense tasks, spatial conditioning | Retrieval, zero-shot |

**For spatial conditioning in DiT**: DINOv3 is superior (better spatial understanding).

### DINOv3 vs. DINOv2

**Key Improvements in v3**:
- RoPE positional encoding (better for variable sizes)
- Register tokens (cleaner attention maps)
- Larger scale (7B model, 1.7B images)
- Better dense task performance

**Migration**: Minimal code changes, mostly drop-in replacement.

---

## References & Resources

**Papers**:
- DINOv3: [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- DINOv2: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- RoFormer (RoPE): [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Code Repositories**:
- Official DINOv3: [github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Hugging Face Transformers: [huggingface.co/docs/transformers/model_doc/dinov3](https://huggingface.co/docs/transformers/en/model_doc/dinov3)

**Model Checkpoints**:
- Hugging Face Hub: [huggingface.co/collections/facebook/dinov3](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009)

**Key Files in Source**:
- `dinov3/models/vision_transformer.py`: Main ViT implementation
- `dinov3/layers/rope_position_encoding.py`: RoPE implementation
- `dinov3/layers/patch_embed.py`: Patch embedding layer

---

## Changelog

**2026-02-20**: Initial documentation
- Discovered RoPE architecture (not fixed-grid)
- Identified rounding bug (14 vs 16)
- Documented variable-length sequence support
- Added integration best practices

---

## License

DINOv3 is released under the [DINOv3 License Agreement](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).

This documentation is MIT licensed.

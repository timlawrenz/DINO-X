# Research: DINOv3 Training Warm-up Schedules

## Findings
Research into training schedules for recent Vision Transformers (ViT) and DINO-family models (DINOv2, DINOv3-derived works) suggests a consensus on learning rate scheduling to ensure stability, especially for large models.

### Standard Pattern
The dominant pattern is **Linear Warm-up followed by Cosine Decay**.

1.  **Linear Warm-up**: The learning rate increases linearly from a very small value (essentially 0) to the target base learning rate (e.g., `1e-4` or `5e-4`) over a fixed number of iterations or epochs.
    *   **Duration**: Typically the first **10-15% of training** or a fixed number of steps (e.g., **3,000 steps** for large datasets).
    *   **Purpose**: Stabilizes the early gradients when the model weights are random, preventing divergence or early feature collapse (a critical risk in DINO).

2.  **Cosine Decay**: After the warm-up, the learning rate follows a cosine curve, decaying from the base learning rate down to a minimum learning rate.
    *   **Target**: Decays to a `min_lr`, often `1e-6` or roughly 1-5% of the base LR.
    *   **Purpose**: Allows the model to settle into sharper minima towards the end of training.

### Specific References
*   **E-RayZer (DINO-based)**: Uses a **3,000-iteration linear warm-up** to a peak LR of `4e-4`, followed by cosine decay.
*   **RecTok**: Linear warm-up for the first **50 epochs**, then cosine decay.
*   **DINOv2**: Typically uses a warm-up period followed by a cosine schedule.

## Recommendation for DINO-X
Given our "High-Capacity, High-Latency" strategy on the AMD Strix Halo, stability is paramount. We should adopt the **Linear Warm-up + Cosine Decay** schedule.

### Configuration
We will add the following parameters to the `phase5_big_run.py` script:

*   `--warmup-steps`: Number of steps for linear warm-up.
    *   *Default*: `2500` (Approx. consistent with the 3k finding, adjusted for our batch sizes).
*   `--min-lr`: Minimum learning rate at the end of the cosine schedule.
    *   *Default*: `1e-6`.

### Logic
The learning rate $\eta_t$ at step $t$ given total steps $T$ and warm-up steps $W$:

1.  **If $t < W$**:
    $$ \eta_t = \eta_{base} \times \frac{t}{W} $$ 
2.  **If $t \ge W$**:
    $$ \eta_t = \eta_{min} + 0.5 (\eta_{base} - \eta_{min}) \left(1 + \cos\left(\pi \frac{t - W}{T - W}\right)\right) $$ 

**Note on "Unlimited" Runs**:
If `--max-steps` is not provided (unlimited run), cosine decay is ill-defined because $T$ is unknown. In this case, we will default to **Linear Warm-up followed by Constant LR** (or decay over a very large hypothetical horizon, essentially constant).

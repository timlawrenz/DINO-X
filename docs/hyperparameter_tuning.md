# Hyperparameter Optimization (HPO) Strategy for DINO-X

As we move from "making it run" to "making it perform," manual tuning becomes inefficient. The search space for Self-Supervised Learning (SSL) involves complex interactions between temperature, momentum, learning rate, and batch size.

This document outlines systematic approaches to explore this "universe" of hyperparameters.

## 1. The "Bandit" Approach (Recommended)
**Method:** **ASHA (Asynchronous Successive Halving Algorithm)** or **Hyperband**.

Instead of guessing one good config, we define a search space and let the algorithm "fail fast."

*   **Logic:** Randomly sample 20 configurations. Run all for 500 steps. Kill the bottom 50% based on Retrieval Ratio. Run the survivors to 1000 steps. Kill bottom 50%. Repeat.
*   **Pros:** Extremely efficient. You don't waste compute on dead-end runs (like our collapsed `center_momentum=0.9` run) because they are pruned early.
*   **Tools:** `Ray Tune`, `Optuna`.

### Proposed Search Space
```python
search_space = {
    "lr": Float(1e-5, 5e-4, log=True),
    "teacher_temp": Float(0.01, 0.07),
    "center_momentum": Float(0.900, 0.999),
    "gram_weight": Float(0.0, 100.0),
    "out_dim": Categorical([2048, 4096, 8192])
}
```

## 2. Population Based Training (PBT)
**Method:** Genetic Algorithm / Evolutionary Strategy.

This is the "Genetic Algorithm" you intuitively identified. It was popularized by DeepMind.

*   **Logic:** Train a population of models in parallel. Periodically (e.g., every 50 epochs), replace the weights of the bottom 20% performers with the weights of the top 20%. **Mutate** the hyperparameters of the clones (e.g., multiply LR by 0.8 or 1.2).
*   **Pros:** Discovers **dynamic** schedules. It might "discover" warmup or cosine decay naturally. It saves training time because you don't restart from scratch; you evolve *during* training.
*   **Cons:** Requires parallel hardware. You need 8-16 GPUs running simultaneously to maintain a healthy population.

## 3. $\mu$-Parametrization ($\mu$P)
**Method:** Zero-shot Transfer from Small to Large.

Developed by Microsoft, this is the most scientifically rigorous method for scaling.

*   **Logic:** Standard PyTorch init/learning rates do not scale consistently with width. $\mu$P changes the initialization and LR scaling rules so that **hyperparameters found on a tiny model (width=128) are optimal for a giant model (width=4096).**
*   **Workflow:**
    1.  Create a tiny `vit-nano` (width=64).
    2.  Brute-force grid search optimal LR, Temp, Momentum on `vit-nano` (takes minutes).
    3.  Copy those parameters *exactly* to `vit-giant`.
*   **Pros:** Solves the "Big Run" risk. You know the parameters will work before you burn 1000 GPU hours.

## 4. Bayesian Optimization
**Method:** Gaussian Processes / TPE (Tree-structured Parzen Estimator).

*   **Logic:** Build a probabilistic model of `Loss = f(Params)`. Based on previous runs, predict which parameters maximize the probability of improvement (Expected Improvement).
*   **Pros:** Very sample efficient. Good for finding "sharp" global minima that random search misses.
*   **Cons:** Hard to parallelize; standard implementations assume the objective function is static (doesn't handle early stopping/pruning as naturally as Bandits).

## Summary Recommendation

1.  **Immediate Step:** Implement **Optuna** with **ASHA Pruning**.
    *   This works on a single node.
    *   It automates the "run for 1000 steps, check retrieval, kill if bad" loop we have been doing manually.
    
2.  **Long-term Step:** Adopt **$\mu$P** rules for the architecture.
    *   This effectively "solves" the Learning Rate and Initialization questions permanently, allowing us to focus on the objective function (Gram vs KoLeo vs DINO).

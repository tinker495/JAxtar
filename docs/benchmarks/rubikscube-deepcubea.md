# Rubik's Cube
- Size: 3x3x3
- Test Set: DeepCubeA Test Set (1000 instances)

### Model Configurations

The following model configurations are used throughout the benchmark results. Each model represents a different training approach and parameter scale, with specific optimizations for Rubik's Cube solving:

- **14.7M params**: Standard DeepCubeA Model Parameters
  The baseline model trained using the original DeepCubeA methodology, containing approximately 14.7 million parameters. This high-capacity neural network architecture is specifically designed for accurate Rubik's Cube state evaluation and provides a strong foundation for comparison with more advanced training techniques.

- **4M params**: 4M CayleyPy Model Parameters
  A more compact version of the CayleyPy Paper's model with approximately 4 million parameters. This reduced-parameter model maintains competitive performance while requiring less memory and computational resources, making it more suitable for deployment on resource-constrained hardware.

- **davi_lt01**: DAVI loss thresholded(0.1) - 14.7M params
  A 14.7M parameter model trained using the DAVI (Deep Approximate Value Iteration) algorithm with a loss threshold of 0.1. This technique selectively targets training updates on samples where the prediction error exceeds the threshold, effectively concentrating learning on more challenging states. The DAVI framework adapts value iteration principles to neural distance estimation, providing robust heuristic guidance for search algorithms.

- **qlearning_lt01**: Q-learning loss thresholded(0.1) - 14.7M params
  A 14.7M parameter Q-learning model trained with loss thresholding at 0.1. By filtering training samples based on temporal difference error magnitude, this approach prioritizes learning from high-uncertainty states, enhancing the model's ability to generalize across the Rubik's Cube state space. The Q-function representation enables direct action-value estimation, making it particularly effective for action-selection algorithms such as Q* and Q-Beam search that depend on precise Q-value predictions for optimal decision making.

- **(int8)**: AQT int8 post-training quantization
  Rows marked with `(int8)` use the same trained checkpoint as their base row, served with AQT int8 quantization (`--use-quantize`). The pretrained weights are converted to int8 once at load time (no retraining), and dense layers run as int8 GEMMs with quantized activations. This roughly halves model memory and speeds up large-batch heuristic evaluation, at the cost of a small heuristic accuracy loss (slightly lower R²/CCC and optimal rate).

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- Software: JAX 0.8.1 / JAxtar `11c11a51` / PuXle `2927cb6` / Xtructure `8f98e3a`
- Avg Nodes is `generated_size`: hash-table usage for eager A* / Beam, and expansions × branching factor for the deferred family (A* Deferred, Q*), so the column is comparable across both families
- DAVI / Q-learning rows are from the previous run (JAxtar `97d1bc72` / PuXle `b522d057` / Xtructure `b1844d4c`); timings are not directly comparable to the diffusion rows.

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | DAVI loss thresholded(0.1) - 14.7M params | 99.80% / 70.24% | 6.812s | 7.67M | 21.23 | 0.797 | 0.872 | Use 50M max nodes |
| A* | Diffusion Distance - 14.7M params | 100% / 58.00% | 0.990s | 1.81M | 21.53 | 0.897 | 0.954 | |
| A* | Diffusion Distance - 14.7M params(int8) | 100% / 57.20% | 0.756s | 1.81M | 21.54 | 0.894 | 0.953 | |
| A* | Diffusion Distance - 4M params | 100% / 47.70% | 0.379s | 1.84M | 21.77 | 0.886 | 0.949 | |
| A* | Diffusion Distance - 4M params(int8) | 100% / 46.00% | 0.319s | 1.84M | 21.82 | 0.887 | 0.949 | |
| A* Deferred | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 70.20% | 7.362s | 764K | 21.24 | 0.797 | 0.872 | |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 57.90% | 0.972s | 2.05M | 21.53 | 0.897 | 0.954 | |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% / 57.30% | 0.742s | 2.05M | 21.54 | 0.894 | 0.953 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 47.70% | 0.436s | 2.08M | 21.77 | 0.886 | 0.949 | |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% / 46.10% | 0.376s | 2.08M | 21.82 | 0.887 | 0.949 | |
| Q* | Q-learning loss thresholded(0.1) - 14.7M params | 100% / 28.60% | 1.236s | 264K | 22.39 | 0.871 | 0.923 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 52.10% | 0.248s | 2.06M | 21.65 | 0.918 | 0.962 | |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% / 52.00% | 0.232s | 2.06M | 21.66 | 0.917 | 0.961 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 42.20% | 0.195s | 2.09M | 21.93 | 0.911 | 0.958 | |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% / 40.50% | 0.190s | 2.09M | 21.97 | 0.909 | 0.958 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 57.70% | 0.806s | 187K | 21.54 | 0.897 | 0.954 | |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% / 56.80% | 0.582s | 187K | 21.56 | 0.894 | 0.953 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 47.00% | 0.270s | 189K | 21.80 | 0.886 | 0.949 | |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% / 46.50% | 0.199s | 189K | 21.81 | 0.887 | 0.949 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 49.40% | 0.084s | 137K | 21.72 | 0.918 | 0.962 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% / 49.40% | 0.066s | 137K | 21.72 | 0.917 | 0.961 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 37.80% | 0.074s | 138K | 22.06 | 0.911 | 0.958 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% / 38.30% | 0.073s | 138K | 22.04 | 0.909 | 0.958 | |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 87.40% | 7.903s | 4.26M | 20.89 | 0.898 | 0.954 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 80.50% | 3.941s | 4.3M | 21.03 | 0.883 | 0.948 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 84.20% | 1.254s | 4.28M | 20.95 | 0.941 | 0.970 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 76.90% | 0.850s | 4.32M | 21.10 | 0.932 | 0.965 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 85.50% | 8.052s | 4.28M | 20.93 | 0.894 | 0.953 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 80.00% | 3.649s | 4.31M | 21.04 | 0.883 | 0.948 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 81.30% | 0.851s | 2.98M | 21.01 | 0.941 | 0.970 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 72.30% | 0.452s | 3.07M | 21.20 | 0.932 | 0.965 | |

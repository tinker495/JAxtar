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
- Software: JAX 0.8.1 / JAxtar `0e8054df` / PuXle `c07dc28` / Xtructure `7a6598a`
- DAVI / Q-learning rows are from the previous run (JAxtar `97d1bc72` / PuXle `b522d057` / Xtructure `b1844d4c`); timings are not directly comparable to the diffusion rows.

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | DAVI loss thresholded(0.1) - 14.7M params | 99.80% / 70.24% | 6.812s | 7.67M | 21.23 | 0.797 | 0.872 | Use 50M max nodes |
| A* | Diffusion Distance - 14.7M params | 100% / 58.40% | 1.217s | 1.81M | 21.52 | 0.897 | 0.954 | |
| A* | Diffusion Distance - 14.7M params(int8) | 100% / 57.20% | 0.948s | 1.81M | 21.55 | 0.894 | 0.953 | |
| A* | Diffusion Distance - 4M params | 100% / 47.60% | 0.630s | 1.84M | 21.78 | 0.886 | 0.949 | |
| A* | Diffusion Distance - 4M params(int8) | 100% / 46.90% | 0.547s | 1.84M | 21.80 | 0.887 | 0.949 | |
| A* Deferred | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 70.20% | 7.362s | 764K | 21.24 | 0.797 | 0.872 | |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 58.00% | 1.169s | 186K | 21.53 | 0.897 | 0.954 | |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% / 57.00% | 0.922s | 187K | 21.56 | 0.894 | 0.953 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 47.90% | 0.533s | 189K | 21.77 | 0.886 | 0.949 | |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% / 46.20% | 0.471s | 189K | 21.82 | 0.887 | 0.949 | |
| Q* | Q-learning loss thresholded(0.1) - 14.7M params | 100% / 28.60% | 1.236s | 264K | 22.39 | 0.871 | 0.923 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 52.10% | 0.365s | 187K | 21.65 | 0.918 | 0.962 | |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% / 52.60% | 0.342s | 187K | 21.64 | 0.917 | 0.961 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 42.40% | 0.320s | 190K | 21.93 | 0.911 | 0.958 | |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% / 40.50% | 0.304s | 190K | 21.97 | 0.909 | 0.958 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 57.30% | 0.837s | 187K | 21.55 | 0.897 | 0.954 | |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% / 56.40% | 0.571s | 187K | 21.57 | 0.894 | 0.953 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 46.50% | 0.271s | 189K | 21.82 | 0.886 | 0.949 | |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% / 44.90% | 0.205s | 190K | 21.85 | 0.887 | 0.949 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 49.40% | 0.084s | 145K | 21.73 | 0.918 | 0.962 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% / 48.70% | 0.065s | 145K | 21.74 | 0.917 | 0.961 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 37.10% | 0.073s | 146K | 22.08 | 0.911 | 0.958 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% / 37.50% | 0.067s | 147K | 22.08 | 0.909 | 0.958 | |

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

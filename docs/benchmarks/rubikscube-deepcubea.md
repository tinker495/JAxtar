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

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | DAVI loss thresholded(0.1) - 14.7M params | 99.80% / 70.24% | 6.812s | 7.67M | 21.23 | 0.797 | 0.872 | Use 50M max nodes |
| A* | Diffusion Distance - 14.7M params | 100% / 54.60% | 1.438s | 1.82M | 21.61 | 0.891 | 0.952 | |
| A* | Diffusion Distance - 4M params | 100% / 45.70% | 0.973s | 1.84M | 21.86 | 0.883 | 0.948 | |
| A* Deferred | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 70.20% | 7.362s | 764K | 21.24 | 0.797 | 0.872 | |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 54.50% | 1.527s | 187K | 21.61 | 0.891 | 0.952 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 45.70% | 0.973s | 1.84M | 21.86 | 0.883 | 0.948 | |
| Q* | Q-learning loss thresholded(0.1) - 14.7M params | 100% / 28.60% | 1.236s | 264K | 22.39 | 0.871 | 0.923 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 51.30% | 0.710s | 188K | 21.68 | 0.941 | 0.970 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 39.40% | 0.699s | 191K | 22.01 | 0.932 | 0.965 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 53.70% | 0.943s | 187K | 21.63 | 0.891 | 0.952 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 44.70% | 0.468s | 190K | 21.88 | 0.883 | 0.948 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 47.70% | 0.094s | 130K | 21.77 | 0.941 | 0.970 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 34.50% | 0.055s | 134K | 22.15 | 0.932 | 0.965 | |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 86.00% | 8.350s | 4.27M | 20.92 | 0.894 | 0.953 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 80.50% | 3.941s | 4.3M | 21.03 | 0.883 | 0.948 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 84.20% | 1.254s | 4.28M | 20.95 | 0.941 | 0.970 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 76.90% | 0.850s | 4.32M | 21.10 | 0.932 | 0.965 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 85.50% | 8.052s | 4.28M | 20.93 | 0.894 | 0.953 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 80.00% | 3.649s | 4.31M | 21.04 | 0.883 | 0.948 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 81.30% | 0.851s | 2.98M | 21.01 | 0.941 | 0.970 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 72.30% | 0.452s | 3.07M | 21.20 | 0.932 | 0.965 | |

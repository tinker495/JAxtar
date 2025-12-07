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
| A* | Diffusion Distance - 14.7M params | 100% / 51.80% | 1.413s | 1.82M | 21.68 | 0.893 | 0.952 | |
| A* | Diffusion Distance - 4M params | 100% / 45.70% | 0.947s | 1.84M | 21.84 | 0.883 | 0.948 | |
| A* Deferred | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 70.20% | 7.362s | 764K | 21.24 | 0.797 | 0.872 | |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 51.80% | 1.508s | 188K | 21.68 | 0.893 | 0.952 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 45.70% | 0.937s | 189K | 21.84 | 0.883 | 0.948 | |
| Q* | Q-learning loss thresholded(0.1) - 14.7M params | 100.00% / 28.60% | 1.236s | 264K | 22.39 | 0.871 | 0.923 | |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 49.50% | 0.589s | 188K | 21.72 | 0.87 | 0.945 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 36.50% | 0.571s | 192K | 22.1 | 0.929 | 0.963 | |
| Beam Search | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 33.50% | 1.098s | 194K | 22.24 | 0.797 | 0.872 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 51.10% | 1.078s | 188K | 21.70 | 0.893 | 0.952 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 44% | 0.595s | 190K | 21.89 | 0.883 | 0.948 | |
| Q-Beam Search | Q-learning loss thresholded(0.1) - 14.7M params | 100.00% / 25.90% | 0.293s | 136K | 22.51 | 0.871 | 0.923 | |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 46.30% | 0.278s | 137K | 21.8 | 0.87 | 0.945 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 33.90% | 0.234s | 134K | 22.21 | 0.929 | 0.963 | |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | DAVI loss thresholded(0.1) - 14.7M params | 100% / 75.70% | 11.357s | 4.41M | 21.12 | 0.797 | 0.872 | |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 84.40% | 9.088s | 4.28M | 20.95 | 0.892 | 0.952 | |
| A* Deferred | Diffusion Distance - 4M params | 100% / 80.7% | 4.383s | 4.3M | 21.02 | 0.882 | 0.948 | |
| Q* | Q-learning loss thresholded(0.1) - 14.7M params | 100% / 81.50% | 1.278s | 4.29M | 21.01 | 0.87 | 0.945 | Not yet |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 81.50% | 1.278s | 4.29M | 21.01 | 0.87 | 0.945 | |
| Q* | Diffusion Distance warmup - 4M params | 100% / 73.7.% | 0.895s | 4.34M | 21.17 | 0.929 | 0.963 | |
| Beam Search | DAVI loss thresholded(0.1) - 14.7M params | 100.00% / 73.20% | 9.781s | 4.35M | 21.18 | 0.797 | 0.872 | |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 84.10% | 7.878s | 4.29M | 20.96 | 0.892 | 0.952 | |
| Beam Search | Diffusion Distance - 4M params | 100% / 80.3% | 3.669s | 4.31M | 21.03 | 0.882 | 0.948 | |
| Q-Beam Search | Q-learning loss thresholded(0.1) - 14.7M params | 100% / 84.10% | 7.878s | 4.29M | 20.96 | 0.892 | 0.952 | Not yet |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 79.80% | 0.846s | 3.14M | 21.04 | 0.87 | 0.945 | |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 70.60% | 0.453s | 3.06M | 21.24 | 0.929 | 0.963 | |

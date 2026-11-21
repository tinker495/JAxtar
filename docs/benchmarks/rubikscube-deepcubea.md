# Rubik's Cube — DeepCubeA Benchmark

- Size: 3x3x3 (QTM metric)
- Test Set: DeepCubeA Test Set (1000 instances)
- Configuration: Batch Size `10K` / Max Node Size `20M` / Cost Weight `0.6` / Pop Ratio `inf`
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- Software: JAX `0.11.2` / JAxtar `903d5cd6` / PuXle `bdeab19` / Xtructure `bb0c0b9`

## Architectural and Measurement Notes

- **Cross-family node count (`Avg Nodes`)**: Reports `generated_size`, which is hash-table occupancy for eager search (`A*`, `Beam`) and `expansions × branching factor` (`estimated_successors`) for deferred search (`A* Deferred`, `Q*`), making search effort comparable across both families.
- **AQT `int8` quantization (`(int8)`)**: Converts pretrained checkpoints once at load time (`--use-quantize`) without retraining; dense layers run as `int8` GEMMs with quantized activations, roughly halving model memory and speeding up large-batch inference at the cost of a small accuracy loss (`R²`/`CCC` and optimal rate).
- **Unquantized-head AQT constraint**: Under `int8` serving, the final action head remains unquantized because `cuBLASLt` `int8` GEMMs require 4-aligned dimensions and non-4-aligned `action_size` (such as 18) crashes compilation on GPUs without Triton fallback.
- **Large-batch eager memory (`batch_size=2^18`)**: Eager `A*` materializes and evaluates all `action_size × batch_size` child states simultaneously, causing OOM at `batch_size=2^18` where deferred `A*`/`Q*` and `Beam` keep peak evaluation memory bounded.

## Results (DeepCubeA Configuration)

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% / 57.90% | 0.857s | 1.81M | 21.53 | 0.897 | 0.954 |
| A* | Diffusion Distance - 14.7M params(int8) | 100% / 57.20% | 0.636s | 1.81M | 21.54 | 0.894 | 0.953 |
| A* | Diffusion Distance - 4M params | 100% / 47.70% | 0.296s | 1.84M | 21.77 | 0.886 | 0.949 |
| A* | Diffusion Distance - 4M params(int8) | 100% / 47.70% | 0.235s | 1.84M | 21.78 | 0.887 | 0.949 |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 57.90% | 0.904s | 2.05M | 21.53 | 0.897 | 0.954 |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% / 57.20% | 0.637s | 2.05M | 21.54 | 0.894 | 0.953 |
| A* Deferred | Diffusion Distance - 4M params | 100% / 47.70% | 0.307s | 2.08M | 21.77 | 0.886 | 0.949 |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% / 47.60% | 0.240s | 2.08M | 21.78 | 0.887 | 0.949 |
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 52.20% | 0.159s | 2.06M | 21.65 | 0.918 | 0.962 |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% / 52.00% | 0.135s | 2.06M | 21.66 | 0.917 | 0.961 |
| Q* | Diffusion Distance warmup - 4M params | 100% / 42.20% | 0.103s | 2.09M | 21.93 | 0.911 | 0.958 |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% / 40.20% | 0.099s | 2.09M | 21.97 | 0.909 | 0.958 |
| Beam Search | Diffusion Distance - 14.7M params | 100% / 57.50% | 0.800s | 187K | 21.55 | 0.897 | 0.954 |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% / 56.90% | 0.559s | 187K | 21.56 | 0.894 | 0.953 |
| Beam Search | Diffusion Distance - 4M params | 100% / 47.60% | 0.228s | 189K | 21.78 | 0.886 | 0.949 |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% / 46.60% | 0.176s | 189K | 21.81 | 0.887 | 0.949 |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 49.40% | 0.095s | 137K | 21.72 | 0.918 | 0.962 |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% / 48.90% | 0.061s | 137K | 21.73 | 0.917 | 0.961 |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 37.80% | 0.028s | 138K | 22.06 | 0.911 | 0.958 |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% / 38.30% | 0.024s | 138K | 22.04 | 0.909 | 0.958 |

# Rubik's Cube — Santa333 Benchmark

- Size: 3x3x3 (UQTM metric)
- Test Set: Santa333 Test Set (82 instances)
- Configuration: Batch Size `10K` / Max Node Size `20M` / Cost Weight `0.6` / Pop Ratio `inf`
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- Software: JAX `0.11.2` / JAxtar `903d5cd6` / PuXle `bdeab19` / Xtructure `bb0c0b9`

See [DeepCubeA Benchmark Notes](./rubikscube-deepcubea.md#architectural-and-measurement-notes) for `generated_size` cross-family comparability, AQT `int8` quantization tradeoffs, the `cuBLASLt` unquantized-head constraint (`int8` GEMMs require 4-aligned dims, whereas non-4-aligned action heads crash compilation without Triton fallback), and eager `batch_size=2^18` OOM behavior.

## Results (DeepCubeA Configuration)

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% | 1.290s | 2.6M | 20.27 | 0.937 | 0.972 |
| A* | Diffusion Distance - 14.7M params(int8) | 100% | 0.918s | 2.6M | 20.29 | 0.943 | 0.974 |
| A* | Diffusion Distance - 4M params | 100% | 0.426s | 2.67M | 20.66 | 0.948 | 0.976 |
| A* | Diffusion Distance - 4M params(int8) | 100% | 0.336s | 2.68M | 20.71 | 0.951 | 0.977 |
| A* Deferred | Diffusion Distance - 14.7M params | 100% | 1.237s | 3.01M | 20.29 | 0.939 | 0.972 |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% | 0.912s | 3.01M | 20.29 | 0.943 | 0.974 |
| A* Deferred | Diffusion Distance - 4M params | 100% | 0.416s | 3.07M | 20.66 | 0.948 | 0.976 |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% | 0.330s | 3.07M | 20.71 | 0.950 | 0.977 |
| Q* | Diffusion Distance warmup - 14.7M params | 100% | 0.175s | 3.04M | 20.49 | 0.940 | 0.973 |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% | 0.157s | 3.04M | 20.51 | 0.942 | 0.973 |
| Q* | Diffusion Distance warmup - 4M params | 100% | 0.114s | 3.11M | 20.90 | 0.940 | 0.971 |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% | 0.111s | 3.13M | 21.02 | 0.941 | 0.971 |
| Beam Search | Diffusion Distance - 14.7M params | 100% | 1.225s | 177K | 20.27 | 0.942 | 0.974 |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% | 0.877s | 177K | 20.29 | 0.947 | 0.976 |
| Beam Search | Diffusion Distance - 4M params | 100% | 0.350s | 181K | 20.66 | 0.953 | 0.978 |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% | 0.265s | 181K | 20.71 | 0.955 | 0.979 |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% | 0.080s | 133K | 20.68 | 0.455 | 0.742 |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% | 0.059s | 133K | 20.71 | 0.454 | 0.741 |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% | 0.041s | 136K | 21.10 | 0.460 | 0.736 |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% | 0.024s | 137K | 21.17 | 0.459 | 0.734 |

# Rubik's Cube
- Size: 3x3x3
- Test Set: Santa333 Test Set (82 instances)
- `(int8)` rows: AQT int8 post-training quantization via `--use-quantize` (see Model Configurations in `rubikscube-deepcubea.md`)

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- Software: JAX 0.8.1 / JAxtar `22ca3bc0` / PuXle `e50627c` / Xtructure `3a5953a`

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% | 1.655 | 2.61M | 20.32 | 0.940 | 0.973 | - |
| A* | Diffusion Distance - 14.7M params(int8) | 100% | 1.292 | 2.61M | 20.32 | 0.942 | 0.974 | - |
| A* | Diffusion Distance - 4M params | 100% | 0.878 | 2.66M | 20.63 | 0.948 | 0.976 | - |
| A* | Diffusion Distance - 4M params(int8) | 100% | 0.856 | 2.67M | 20.63 | 0.948 | 0.975 | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% | 1.671 | 177K | 20.29 | 0.939 | 0.973 | - |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% | 1.417 | 177K | 20.29 | 0.942 | 0.974 | - |
| A* Deferred | Diffusion Distance - 4M params | 100% | 0.831 | 180K | 20.66 | 0.948 | 0.976 | - |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% | 0.781 | 180K | 20.63 | 0.950 | 0.976 | - |
| Q* | Diffusion Distance warmup - 14.7M params | 100% | 0.473 | 179K | 20.46 | 0.939 | 0.972 | - |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% | 0.471 | 179K | 20.49 | 0.941 | 0.973 | - |
| Q* | Diffusion Distance warmup - 4M params | 100% | 0.519 | 183K | 20.90 | 0.940 | 0.971 | - |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% | 0.490 | 184K | 20.98 | 0.942 | 0.972 | - |
| Beam Search | Diffusion Distance - 14.7M params | 100% | 1.265 | 177K | 20.34 | 0.945 | 0.975 | - |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% | 1.050 | 178K | 20.37 | 0.948 | 0.976 | - |
| Beam Search | Diffusion Distance - 4M params | 100% | 0.449 | 181K | 20.66 | 0.953 | 0.978 | - |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% | 0.368 | 181K | 20.71 | 0.952 | 0.977 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% | 0.086 | 138K | 20.71 | 0.455 | 0.742 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% | 0.067 | 138K | 20.68 | 0.454 | 0.742 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% | 0.054 | 142K | 21.12 | 0.461 | 0.736 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% | 0.051 | 143K | 21.17 | 0.459 | 0.734 | - |
서
# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU
- Software: JAX 0.8.1
- `(int8)` rows require the unquantized-head AQT fix: cuBLASLt has no int8 kernel for the non-4-aligned action-head GEMM on this GPU, crashing compilation otherwise

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% | 11.868s | 4.14M | 19.61 | 0.844 | 0.940 | - |
| A* Deferred | Diffusion Distance - 14.7M params(int8) | 100% | 9.835s | 4.14M | 19.61 | 0.848 | 0.941 | - |
| A* Deferred | Diffusion Distance - 4M params | 100% | 4.308s | 4.18M | 19.76 | 0.873 | 0.949 | - |
| A* Deferred | Diffusion Distance - 4M params(int8) | 100% | 3.424s | 4.17M | 19.71 | 0.864 | 0.946 | - |
| Q* | Diffusion Distance warmup - 14.7M params | 100% | 1.281s | 4.16M | 19.68 | 0.860 | 0.945 | - |
| Q* | Diffusion Distance warmup - 14.7M params(int8) | 100% | 1.156s | 4.16M | 19.68 | 0.847 | 0.940 | - |
| Q* | Diffusion Distance warmup - 4M params | 100% | 0.836s | 4.24M | 20.00 | 0.897 | 0.957 | - |
| Q* | Diffusion Distance warmup - 4M params(int8) | 100% | 0.787s | 4.23M | 19.98 | 0.891 | 0.954 | - |
| Beam Search | Diffusion Distance - 14.7M params | 100% | 11.487s | 4.15M | 19.61 | 0.860 | 0.945 | - |
| Beam Search | Diffusion Distance - 14.7M params(int8) | 100% | 9.464s | 4.15M | 19.61 | 0.863 | 0.945 | - |
| Beam Search | Diffusion Distance - 4M params | 100% | 3.921s | 4.19M | 19.76 | 0.885 | 0.953 | - |
| Beam Search | Diffusion Distance - 4M params(int8) | 100% | 3.041s | 4.17M | 19.71 | 0.879 | 0.950 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% | 0.732s | 3.20M | 19.68 | 0.361 | 0.736 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params(int8) | 100% | 0.608s | 3.22M | 19.73 | 0.369 | 0.736 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% | 0.266s | 3.28M | 20.02 | 0.404 | 0.739 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params(int8) | 100% | 0.217s | 3.30M | 20.07 | 0.411 | 0.740 | - |

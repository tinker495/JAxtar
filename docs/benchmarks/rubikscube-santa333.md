# Rubik's Cube
- Size: 3x3x3
- Test Set: Santa333 Test Set (82 instances)

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% | 2.189 | 2.61M | 20.34 | 0.938 | 0.972 | - |
| A* | Diffusion Distance - 4M params | 100% | 1.587 | 2.64M | 20.49 | 0.945 | 0.974 | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% | 2.090 | 177K | 20.34 | 0.938 | 0.972 | - |
| A* Deferred | Diffusion Distance - 4M params | 100% | 1.408 | 179K | 20.49 | 0.945 | 0.974 | - |
| Q* | Diffusion Distance warmup - 14.7M params | 100% | 0.853 | 179K | 20.54 | 0.951 | 0.974 | - |
| Q* | Diffusion Distance warmup - 4M params | 100% | 0.53 | 184K | 21.00 | 0.920 | 0.955 | - |
| Beam Search | Diffusion Distance - 14.7M params | 100% | 1.319 | 178K | 20.39 | 0.943 | 0.974 | - |
| Beam Search | Diffusion Distance - 4M params | 100% | 0.604 | 179K | 20.54 | 0.951 | 0.977 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% | 0.127 | 127K | 20.83 | 0.458 | 0.712 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% | 0.050 | 130K | 21.10 | 0.437 | 0.693 | - |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% | 12.310s | 4.13M | 19.59 | 0.852 | 0.942 | - |
| A* Deferred | Diffusion Distance - 4M params | 100% | 5.765s | 4.15M | 19.63 | 0.866 | 0.947 | - |
| Q* | Diffusion Distance warmup - 14.7M params | 100% | 1.560s | 4.16M | 19.71 | 0.963 | 0.983 | - |
| Q* | Diffusion Distance warmup - 4M params | 100% | 1.174s | 4.21M | 19.85 | 0.962 | 0.981 | - |
| Beam Search | Diffusion Distance - 14.7M params | 100% | 11.992s | 4.14M | 19.59 | 0.867 | 0.947 | - |
| Beam Search | Diffusion Distance - 4M params | 100% | 5.426s | 4.15M | 19.63 | 0.881 | 0.952 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% | 0.888s | 2.92M | 19.71 | 0.446 | 0.732 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% | 0.490s | 2.99M | 19.95 | 0.454 | 0.725 | - |

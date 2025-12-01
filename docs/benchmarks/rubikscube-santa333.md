# Rubik's Cube
- Size: 3x3x3
- Test Set: Santa333 Test Set (82 instances)

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100.00% | 1.74 | 2.63M | 20.44 | 0.945 | 0.973 | - |
| A* | Diffusion Distance - 4M params | 100.00% | 1.11 | 2.7M | 20.78 | 0.918 | 0.959 | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100.00% | 1.79 | 178K | 20.44 | 0.933 | 0.970 | - |
| A* Deferred | Diffusion Distance - 4M params | 100.00% | 1.12 | 182K | 20.78 | 0.904 | 0.954 | - |
| Q* | Diffusion Distance warmup - 14.7M params | 100.00% | 0.53 | 180K | 20.59 | 0.927 | 0.968 | - |
| Q* | Diffusion Distance warmup - 4M params | 100.00% | 0.53 | 184K | 21.00 | 0.920 | 0.955 | - |
| Beam Search | Diffusion Distance - 14.7M params | 100.00% | 1.27 | 179K | 20.46 | 0.946 | 0.975 | - |
| Beam Search | Diffusion Distance - 4M params | 100.00% | 0.60 | 182K | 20.78 | 0.943 | 0.972 | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100.00% | 0.09 | 132K | 20.68 | 0.442 | 0.746 | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100.00% | 0.07 | 131K | 21.24 | 0.425 | 0.682 | - |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | - | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 4M params | - | - | - | - | - | - | - |
| Q* | Diffusion Distance warmup - 14.7M params | - | - | - | - | - | - | - |
| Q* | Diffusion Distance warmup - 4M params | - | - | - | - | - | - | - |
| Beam Search | Diffusion Distance - 14.7M params | - | - | - | - | - | - | - |
| Beam Search | Diffusion Distance - 4M params | - | - | - | - | - | - | - |
| Q-Beam | Diffusion Distance warmup - 14.7M params | - | - | - | - | - | - | - |
| Q-Beam | Diffusion Distance warmup - 4M params | - | - | - | - | - | - | - |

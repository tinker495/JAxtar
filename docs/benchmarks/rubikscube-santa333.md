# Rubik's Cube
- Size: 3x3x3
- Test Set: Santa333 Test Set (81 instances)

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | - | - | - | - | - | - | - |
| A* | Diffusion Distance - 4M params | - | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | - | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 4M params | - | - | - | - | - | - | - |
| Q* | Diffusion Distance warmup - 14.7M params | - | - | - | - | - | - | - |
| Q* | Diffusion Distance warmup - 4M params | - | - | - | - | - | - | - |
| Beam Search | Diffusion Distance - 14.7M params | - | - | - | - | - | - | - |
| Beam Search | Diffusion Distance - 4M params | - | - | - | - | - | - | - |
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | - | - | - | - | - | - | - |
| Q-Beam Search | Diffusion Distance warmup - 4M params | - | - | - | - | - | - | - |

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
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

# Rubik's Cube
- Size: 3x3x3
- Test Set: DeepCubeA Test Set (1000 instances)

# DeepCubeA Configuration
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% / 51.30% | 1.384s | 1.83M | 21.7 | 0.88 | 0.948 | DeepCubeA Param size|
| A* | Diffusion Distance - 4M params | 100% / 45.70% | 0.947s | 1.84M | 21.84 | 0.883 | 0.948 | CayleyPy Param size|
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 51.30% | 1.466s | 188K | 21.7 | 0.88 | 0.948 | DeepCubeA Param size|
| A* Deferred | Diffusion Distance - 4M params | 100% / 45.70% | 0.937s | 189K | 21.84 | 0.883 | 0.948 | CayleyPy Param size|
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 49.50% | 0.589s | 188K | 21.72 | 0.87 | 0.945 | DeepCubeA Param size|
| Q* | Diffusion Distance warmup - 4M params | 100% / 36.50% | 0.571s | 192K | 22.1 | 0.929 | 0.963 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 14.7M params | 100% / 50.10% | 1.058s | 188K | 21.72 | 0.88 | 0.948 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 4M params | 100% / 44% | 0.595s | 190K | 21.89 | 0.883 | 0.948 | CayleyPy Param size|
| Q-Beam Search | Diffusion Distance warmup - 14.7M params | 100% / 46.30% | 0.278s | 137K | 21.8 | 0.87 | 0.945 | DeepCubeA Param size|
| Q-Beam Search | Diffusion Distance warmup - 4M params | 100% / 33.90% | 0.234s | 134K | 22.21 | 0.929 | 0.963 | CayleyPy Param size|

# CayleyPy Batch Size Configuration
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Hardware: NVIDIA RTX 5090 GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 84.9% | 9.143s | 4.28M | 20.94 | 0.88 | 0.948 | DeepCubeA Param size|
| A* Deferred | Diffusion Distance - 4M params | 100% / 80.7% | 4.383s | 4.3M | 21.02 | 0.882 | 0.948 | CayleyPy Param size|
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 81.50% | 1.278s | 4.29M | 21.01 | 0.87 | 0.945 | DeepCubeA Param size|
| Q* | Diffusion Distance warmup - 4M params | 100% / 73.7.% | 0.895s | 4.34M | 21.17 | 0.929 | 0.963 | CayleyPy Param size|
| Beam Search | Diffusion Distance - 14.7M params | 100% / 84.50% | 7.937s | 4.29M | 20.95 | 0.88 | 0.948 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 4M params | 100% / 80.3% | 3.669s | 4.31M | 21.03 | 0.882 | 0.948 | CayleyPy Param size|
| Q-Beam | Diffusion Distance warmup - 14.7M params | 100% / 79.80% | 0.846s | 3.14M | 21.04 | 0.87 | 0.945 | DeepCubeA Param size|
| Q-Beam | Diffusion Distance warmup - 4M params | 100% / 70.60% | 0.453s | 3.06M | 21.24 | 0.929 | 0.963 | CayleyPy Param size|

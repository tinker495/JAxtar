# Benchmark Results

This document serves as a record of experimental results obtained from running benchmarks on various puzzles.

## Rubik's Cube

**Configuration**:
- Size: 3x3x3
- Test Set: DeepCubeA Test Set (1000 instances)
- Batch Size: 10K / Max Node Size: 20M / Cost Weight: 0.6 / Pop Ratio: inf
- Hardware: NVIDIA RTX 4080Ti GPU

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | Diffusion Distance - 14.7M params | 100% / 51.30% | 1.384s | 1.83M | 21.7 | 0.880 | 0.948 | DeepCubeA Param size|
| A* | Diffusion Distance - 4M params | 100% / 45.70% | 0.947s | 1.84M | 21.84 | 0.883 | 0.948 | CayleyPy Param size|
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 51.30% | 1.466s | 188K | 21.7 | 0.88 | 0.948 | DeepCubeA Param size|
| A* Deferred | Diffusion Distance - 4M params | 100% / 45.70% | 0.937s | 189K | 21.84 | 0.883 | 0.948 | CayleyPy Param size|
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 49.50% | 0.589s | 188K | 21.72 | 0.87 | 0.945 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 14.7M params | 100% / 50.10% | 1.058s | 188K | 21.72 | 0.88 | 0.948 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 4M params | 100% / 44% | 0.595s | 190K | 21.89 | 0.883 | 0.948 | CayleyPy Param size|
| Q-Beam | Diffusion Distance warmup - 14.7M params | 100% / 46.30% | 0.278s | 137K | 21.8 | 0.87 | 0.945 | DeepCubeA Param size|

**Configuration**:
- Batch Size: 2^18 / Max Node Size: 20M / Cost Weight: 0.9 / Pop Ratio: inf
- Note: CayleyPy Batch Size Configuration

| Algorithm | Model | Success Rate / Optimal Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R² (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* | - | Mostly OOM | - | - | - | - | - | - |
| A* Deferred | Diffusion Distance - 14.7M params | 100% / 51.30% | 1.466s | 188K | 21.7 | 0.88 | 0.948 | DeepCubeA Param size|
| A* Deferred | Diffusion Distance - 4M params | 100% / 45.70% | 0.947s | 188K | 21.84 | 0.883 | 0.948 | CayleyPy Param size|
| Q* | Diffusion Distance warmup - 14.7M params | 100% / 49.50% | 0.589s | 188K | 21.72 | 0.87 | 0.945 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 14.7M params | 100% / 50.10% | 1.058s | 188K | 21.72 | 0.88 | 0.948 | DeepCubeA Param size|
| Beam Search | Diffusion Distance - 4M params | 100% / 50.10% | 1.058s | 188K | 21.72 | 0.88 | 0.948 | CayleyPy Param size|
| Q-Beam | Diffusion Distance warmup - 14.7M params | 100% / 46.30% | 0.278s | 137K | 21.8 | 0.87 | 0.945 | DeepCubeA Param size|

---

> **Note**: For details on how to generate these results and interpret the logs, see [Benchmark Logging](./benchmark_logging.md).

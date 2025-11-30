# Benchmark Results

This document serves as a record of experimental results obtained from running benchmarks on various puzzles.

## Rubik's Cube

**Configuration**:
- Size: 3x3x3
- Test Set: DeepCubeA Test Set (1000 instances)

| Algorithm | Model | Success Rate | Avg Time (s) | Avg Nodes | Avg Path Cost | R^2 (Heuristic) | CCC (Heuristic) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A* | - (nn) | - | - | - | - | - | - | |
| A* Deferred | - (nn) | - | - | - | - | - | - | |
| Q* | - (nn) | - | - | - | - | - | - | |
| Beam Search | - (nn) | - | - | - | - | - | - | |
| Q-Beam | - (nn) | - | - | - | - | - | - | |

---

> **Note**: For details on how to generate these results and interpret the logs, see [Benchmark Logging](./benchmark_logging.md).

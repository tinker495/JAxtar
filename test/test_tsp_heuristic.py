import pytest
import jax
import jax.numpy as jnp

from heuristic.tsp_heuristic import TSPHeuristic
from puzzle.tsp import TSP


@pytest.fixture
def random_tsp_instance():
    size = 6  # small for brute-force optimal path enumeration
    puzzle = TSP(size=size)
    key = jax.random.PRNGKey(0)
    solve_config = puzzle.get_solve_config(key=key)
    initial_state = puzzle.get_initial_state(solve_config, key=key)
    return puzzle, solve_config, initial_state


def brute_force_optimal_remaining(puzzle: TSP, solve_config: TSP.SolveConfig, state: TSP.State):
    """Brute-force optimal cost from the given state until tour completion.

    WARNING: factorial complexity, use only for tiny instances (<=7).
    """
    unpacked = state.unpacking()
    visited = unpacked.mask.astype(bool)
    current_idx = int(unpacked.point)
    start_idx = int(solve_config.start)

    n = puzzle.size
    unvisited = [i for i in range(n) if not visited[i]]

    dist = solve_config.distance_matrix

    if not unvisited:
        return float(dist[current_idx, start_idx])

    best = float("inf")

    from itertools import permutations

    for perm in permutations(unvisited):
        cost = dist[current_idx, perm[0]] if perm else 0.0
        for a, b in zip(perm, perm[1:]):
            cost += dist[a, b]
        last_idx = perm[-1] if perm else current_idx
        cost += dist[last_idx, start_idx]
        best = min(best, cost)
    return best


def test_heuristic_admissible(random_tsp_instance):
    puzzle, solve_config, state = random_tsp_instance
    heuristic = TSPHeuristic(puzzle)

    h_val = heuristic.distance(solve_config, state)
    optimal = brute_force_optimal_remaining(puzzle, solve_config, state)

    assert h_val <= optimal + 1e-4, "Heuristic must be admissible (lower-bound)."


def test_heuristic_tighter_than_baseline(random_tsp_instance):
    puzzle, solve_config, state = random_tsp_instance
    heuristic = TSPHeuristic(puzzle)

    # New heuristic value
    h_new = heuristic.distance(solve_config, state)

    # Baseline (old implementation) – mean of pairwise distances among unvisited nodes
    unpacked = state.unpacking()
    visited_mask = unpacked.mask
    inv_mask = 1 - visited_mask
    dmat = solve_config.distance_matrix
    masked = dmat * inv_mask[None, :] * inv_mask[:, None]
    h_old = jnp.mean(jnp.sum(masked, axis=1))

    assert h_new >= h_old - 1e-4, "New heuristic should dominate or equal the old one."
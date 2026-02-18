import jax
import jax.numpy as jnp
import numpy as np
from puxle import SlidePuzzle

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.stars.astar import astar_builder


def test_unified_astar_preserves_valid_parent_chain_for_solved_path():
    """Regression: unified eager A* must keep parent links consistent with unit-step costs."""
    puzzle = SlidePuzzle(size=2)
    heuristic = SlidePuzzleHeuristic(puzzle)
    solve_config, start_state = puzzle.get_inits(jax.random.PRNGKey(0))

    astar = astar_builder(
        puzzle=puzzle,
        heuristic=heuristic,
        batch_size=128,
        max_nodes=50_000,
        cost_weight=0.6,
        show_compile_time=False,
    )
    result = astar(solve_config, start_state)

    assert bool(result.solved)
    solved_cost = int(float(jnp.ravel(result.get_cost(result.solved_idx))[0]))
    assert solved_cost > 0

    solved_path = result.get_solved_path()
    assert len(solved_path) == solved_cost + 1

    parent_indices = np.asarray(result.parent.hashidx.index).astype(np.int64).reshape(-1)
    costs = np.asarray(result.cost).reshape(-1)
    solved_index = int(np.asarray(result.solved_idx.hashidx.index).reshape(-1)[0])

    current_idx = solved_index
    for _ in range(solved_cost):
        assert 0 <= current_idx < parent_indices.shape[0]
        parent_idx = int(parent_indices[current_idx])
        assert 0 <= parent_idx < parent_indices.shape[0]
        assert np.isclose(costs[parent_idx], costs[current_idx] - 1.0)
        current_idx = parent_idx

    assert np.isclose(costs[current_idx], 0.0)

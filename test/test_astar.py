import jax
import jax.numpy as jnp
import pytest

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.astar import astar_builder
from puzzle.slidepuzzle import SlidePuzzle


@pytest.fixture
def astar_setup():
    puzzle_size = 4  # 4x4 puzzle
    batch_size = 128
    max_node_size = 10000
    cost_weight = 1.0 - 1e-3

    puzzle = SlidePuzzle(puzzle_size)
    heuristic = SlidePuzzleHeuristic(puzzle)
    solve_config = puzzle.get_solve_config()

    return {
        "puzzle": puzzle,
        "heuristic": heuristic,
        "batch_size": batch_size,
        "max_node_size": max_node_size,
        "cost_weight": cost_weight,
        "solve_config": solve_config,
    }


def test_astar_initialization(astar_setup):
    setup = astar_setup
    astar_fn = astar_builder(
        setup["puzzle"],
        setup["heuristic"],
        setup["batch_size"],
        setup["max_node_size"],
        cost_weight=setup["cost_weight"],
    )
    assert astar_fn is not None


def test_astar_search(astar_setup):
    setup = astar_setup

    # Build A* search function
    astar_fn = astar_builder(
        setup["puzzle"],
        setup["heuristic"],
        setup["batch_size"],
        setup["max_node_size"],
        cost_weight=setup["cost_weight"],
    )

    # Create initial state
    states = setup["puzzle"].State(
        board=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15], dtype=jnp.uint8)
    )

    # Run search
    search_result = astar_fn(setup["solve_config"], states)

    assert search_result is not None
    assert search_result.solved, "Solution not found"
    assert search_result.solved_idx is not None, "Solved index not found"


def test_heuristic_values(astar_setup):
    setup = astar_setup

    # Create test state
    key = jax.random.PRNGKey(0)
    states = jax.vmap(setup["puzzle"].get_initial_state, in_axes=(None, 0))(
        setup["solve_config"], jax.random.split(key, 1)
    )

    # Calculate heuristic values
    heuristic_values = setup["heuristic"].batched_distance(setup["solve_config"], states)

    assert heuristic_values.shape[0] == 1
    assert jnp.all(heuristic_values >= 0)  # Heuristic should be non-negative

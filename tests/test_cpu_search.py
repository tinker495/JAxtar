"""CPU search specifications requested and approved by tinker495."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from puxle import SlidePuzzle

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.search_build_spec import SearchBuildSpec
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.q_base import QFromHeuristic


# Author: tinker495.
# A search can return normally without solving a puzzle. These CPU cases require
# A*, deferred A*, and Q* to solve the same fixed, non-goal board without model
# downloads. Replaying every returned action checks legal transitions and goal
# arrival independently of the search's solved flag; optimality is not required.
@pytest.mark.parametrize("algorithm", ["astar", "astar-d", "qstar"])
def test_small_puzzle_solution_on_cpu(algorithm: str) -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        puzzle = SlidePuzzle(size=3)
        solve_config = puzzle.get_solve_config()
        start = puzzle.State.from_unpacked(
            board=jnp.array([1, 2, 3, 4, 0, 6, 7, 5, 8], dtype=jnp.uint8)
        )
        assert not bool(puzzle.is_solved(solve_config, start))
        heuristic = SlidePuzzleHeuristic(puzzle)
        spec = SearchBuildSpec(warmup_inputs=(solve_config, start))
        if algorithm == "qstar":
            search = qstar_builder(
                puzzle, QFromHeuristic(heuristic), batch_size=16, max_nodes=256, spec=spec
            )
        else:
            builder = {"astar": astar_builder, "astar-d": astar_d_builder}[algorithm]
            search = builder(puzzle, heuristic, batch_size=16, max_nodes=256, spec=spec)
        result = search(solve_config, start)
        assert {device.platform for device in result.solved.devices()} == {"cpu"}
        assert bool(result.solved)
        trace = result.to_solution_trace(puzzle=puzzle)
        assert trace.solved
        assert trace.actions
        current = start
        for action in trace.actions:
            assert 0 <= action < puzzle.action_size
            current, cost = puzzle.get_actions(solve_config, current, action)
            assert math.isfinite(float(cost)), f"Invalid action {action}"
        assert bool(puzzle.is_solved(solve_config, current))
        np.testing.assert_array_equal(current.board_unpacked, solve_config.GoalSpec.board_unpacked)

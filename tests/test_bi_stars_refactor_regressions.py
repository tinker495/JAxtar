import jax
import jax.numpy as jnp
from puxle import SlidePuzzle
from unittest.mock import patch

from heuristic.empty_heuristic import EmptyHeuristic
from JAxtar.bi_stars.bi_astar import bi_astar_builder
from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder
from JAxtar.bi_stars.bi_qstar import bi_qstar_builder
from qfunction.q_base import QFunction


class _ZeroQFunction(QFunction):
    def q_value(self, q_parameters, current):
        return jnp.zeros((self.puzzle.action_size,), dtype=jnp.float32)


def test_bi_astar_builder_does_not_build_bi_search_result_outside_compile():
    puzzle = SlidePuzzle(size=2)
    heuristic = EmptyHeuristic(puzzle)

    with patch("JAxtar.bi_stars.bi_astar.build_bi_search_result", side_effect=RuntimeError):
        with patch(
            "JAxtar.bi_stars.bi_astar.compile_search_builder",
            side_effect=lambda fn, _puzzle, _show_compile_time, _warmup_inputs: fn,
        ):
            search_fn = bi_astar_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    assert callable(search_fn)


def test_bi_astar_start_equals_goal_stamps_solved_fields():
    puzzle = SlidePuzzle(size=2)
    heuristic = EmptyHeuristic(puzzle)
    search_fn = bi_astar_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    result = search_fn(solve_config, goal)

    assert bool(jax.device_get(result.meeting.found)) is True
    assert bool(jax.device_get(result.forward.solved)) is True
    assert bool(jax.device_get(result.backward.solved)) is True

    assert int(jax.device_get(result.forward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.fwd_hashidx.index)
    )
    assert int(jax.device_get(result.backward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.bwd_hashidx.index)
    )


def test_bi_astar_d_start_equals_goal_stamps_solved_fields():
    puzzle = SlidePuzzle(size=2)
    heuristic = EmptyHeuristic(puzzle)
    search_fn = bi_astar_d_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    result = search_fn(solve_config, goal)

    assert bool(jax.device_get(result.meeting.found)) is True
    assert bool(jax.device_get(result.forward.solved)) is True
    assert bool(jax.device_get(result.backward.solved)) is True

    assert int(jax.device_get(result.forward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.fwd_hashidx.index)
    )
    assert int(jax.device_get(result.backward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.bwd_hashidx.index)
    )


def test_bi_qstar_start_equals_goal_stamps_solved_fields():
    puzzle = SlidePuzzle(size=2)
    q_fn = _ZeroQFunction(puzzle)
    search_fn = bi_qstar_builder(puzzle, q_fn, batch_size=8, max_nodes=64, backward_mode="dijkstra")

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    result = search_fn(solve_config, goal)

    assert bool(jax.device_get(result.meeting.found)) is True
    assert bool(jax.device_get(result.forward.solved)) is True
    assert bool(jax.device_get(result.backward.solved)) is True

    assert int(jax.device_get(result.forward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.fwd_hashidx.index)
    )
    assert int(jax.device_get(result.backward.solved_idx.hashidx.index)) == int(
        jax.device_get(result.meeting.bwd_hashidx.index)
    )


def test_bi_astar_does_not_leak_state_across_invocations():
    puzzle = SlidePuzzle(size=2)
    heuristic = EmptyHeuristic(puzzle)
    search_fn = bi_astar_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start2 = neighbours[action, 0]

    res_goal_1 = search_fn(solve_config, goal)
    res_start2 = search_fn(solve_config, start2)
    res_goal_2 = search_fn(solve_config, goal)

    assert float(jax.device_get(res_goal_1.meeting.total_cost)) == 0.0
    assert float(jax.device_get(res_goal_2.meeting.total_cost)) == 0.0
    assert jnp.isfinite(res_start2.meeting.total_cost)


def test_bi_astar_d_does_not_leak_state_across_invocations():
    puzzle = SlidePuzzle(size=2)
    heuristic = EmptyHeuristic(puzzle)
    search_fn = bi_astar_d_builder(puzzle, heuristic, batch_size=8, max_nodes=64)

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start2 = neighbours[action, 0]

    res_goal_1 = search_fn(solve_config, goal)
    _ = search_fn(solve_config, start2)
    res_goal_2 = search_fn(solve_config, goal)

    assert float(jax.device_get(res_goal_1.meeting.total_cost)) == 0.0
    assert float(jax.device_get(res_goal_2.meeting.total_cost)) == 0.0


def test_bi_qstar_does_not_leak_state_across_invocations():
    puzzle = SlidePuzzle(size=2)
    q_fn = _ZeroQFunction(puzzle)
    search_fn = bi_qstar_builder(puzzle, q_fn, batch_size=8, max_nodes=64, backward_mode="dijkstra")

    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))

    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, jnp.array([True]))
    valid = jnp.isfinite(ncosts[:, 0])
    action = int(jax.device_get(jnp.argmax(valid)))
    start2 = neighbours[action, 0]

    res_goal_1 = search_fn(solve_config, goal)
    _ = search_fn(solve_config, start2)
    res_goal_2 = search_fn(solve_config, goal)

    assert float(jax.device_get(res_goal_1.meeting.total_cost)) == 0.0
    assert float(jax.device_get(res_goal_2.meeting.total_cost)) == 0.0

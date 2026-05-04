import jax
import jax.numpy as jnp
from puxle import SlidePuzzle
from xtructure import HashIdx

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.bi_stars.bi_search_base import (
    MeetingPoint,
    build_bi_search_result,
    initialize_bi_loop_common,
    materialize_meeting_point_hashidxs,
    reconstruct_bidirectional_path,
)


def _make_valid_slide_solve_config(puzzle: SlidePuzzle):
    board = jnp.concatenate(
        [
            jnp.arange(1, puzzle.size**2, dtype=jnp.uint8),
            jnp.array([0], dtype=jnp.uint8),
        ],
        axis=0,
    )
    goal = puzzle.State.default().set_unpacked(board=board)
    return puzzle.SolveConfig(TargetState=goal), goal


def _states_equal(left, right) -> bool:
    equal_leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), left, right)
    )
    return all(bool(jax.device_get(leaf)) for leaf in equal_leaves)


def _find_one_step_start_to_goal(puzzle: SlidePuzzle, solve_config, goal):
    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    filled_b = jnp.array([True])
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, goal_b, filled_b)

    start = None
    for action in range(puzzle.action_size):
        if bool(jax.device_get(jnp.isfinite(ncosts[action, 0]))) and not _states_equal(
            neighbours[action, 0], goal
        ):
            start = neighbours[action, 0]
            break
    assert start is not None

    start_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], start)
    neighbours, ncosts = puzzle.batched_get_neighbours(solve_config, start_b, filled_b)
    for action in range(puzzle.action_size):
        if bool(jax.device_get(jnp.isfinite(ncosts[action, 0]))) and _states_equal(
            neighbours[action, 0], goal
        ):
            return start, action, ncosts[action, 0].astype(KEY_DTYPE)

    raise AssertionError("Expected a one-step path from generated start to goal")


def _find_inverse_action_to_start(puzzle: SlidePuzzle, inverse_solveconfig, goal, start):
    goal_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], goal)
    inv_neighbours, inv_costs = puzzle.batched_get_inverse_neighbours(
        inverse_solveconfig,
        goal_b,
        jnp.array([True]),
    )

    for action in range(puzzle.action_size):
        if bool(jax.device_get(jnp.isfinite(inv_costs[action, 0]))) and _states_equal(
            inv_neighbours[action, 0], start
        ):
            return action, inv_costs[action, 0].astype(KEY_DTYPE)

    raise AssertionError("Expected inverse neighbours to contain the one-step start")


def _assert_path_replays_to_goal(puzzle: SlidePuzzle, solve_config, path, goal):
    assert len(path) >= 2

    state = path[0][1]
    for action, expected_state in path[1:]:
        state_b = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], state)
        next_state, _ = puzzle.batched_get_actions(
            solve_config,
            state_b,
            jnp.array([action], dtype=ACTION_DTYPE),
            jnp.array([True]),
        )
        state = jax.tree_util.tree_map(lambda x: x[0], next_state)
        assert _states_equal(state, expected_state)

    assert _states_equal(state, goal)


def test_import():
    assert build_bi_search_result is not None


def test_build_bi_search_result_initializes_forward_and_backward_consistently():
    puzzle = SlidePuzzle(size=2)
    batch_size = 8
    max_nodes = 32

    result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=batch_size,
        max_nodes=max_nodes,
        action_size=puzzle.action_size,
    )

    assert result.batch_size == batch_size
    assert result.forward.capacity == max_nodes
    assert result.backward.capacity == max_nodes
    assert int(result.forward.generated_size) == 0
    assert int(result.backward.generated_size) == 0

    assert bool(result.meeting.found) is False
    assert jnp.isinf(result.meeting.total_cost)


def test_initialize_bi_loop_common_sets_meeting_when_start_equals_goal():
    puzzle = SlidePuzzle(size=2)
    solve_config = puzzle.SolveConfig.default()
    goal = puzzle.solve_config_to_state_transform(solve_config, key=jax.random.PRNGKey(0))
    start = goal

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    assert bool(jax.device_get(bi_result.meeting.found)) is True
    assert float(jax.device_get(bi_result.meeting.total_cost)) == 0.0
    assert float(jax.device_get(bi_result.meeting.fwd_cost)) == 0.0
    assert float(jax.device_get(bi_result.meeting.bwd_cost)) == 0.0


def test_materialize_meeting_point_hashidxs_materializes_edge_only_meeting():
    puzzle = SlidePuzzle(size=2)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)
    start, action, step_cost = _find_one_step_start_to_goal(puzzle, solve_config, goal)

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    _, fwd_current, _, _, bwd_current, _ = initialize_bi_loop_common(
        bi_result, puzzle, solve_config, start
    )
    start_hashidx = fwd_current.hashidx[0]
    meeting_state = goal
    meeting_bwd_hashidx = bwd_current.hashidx[0]

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)
    bi_result.meeting = MeetingPoint(
        fwd_hashidx=dummy_hashidx,
        bwd_hashidx=meeting_bwd_hashidx,
        fwd_cost=step_cost,
        bwd_cost=jnp.array(0, dtype=KEY_DTYPE),
        total_cost=step_cost,
        found=jnp.array(True),
        fwd_has_hashidx=jnp.array(False),
        bwd_has_hashidx=jnp.array(True),
        fwd_parent_hashidx=start_hashidx,
        fwd_parent_action=jnp.array(action, dtype=ACTION_DTYPE),
        bwd_parent_hashidx=dummy_hashidx,
        bwd_parent_action=dummy_action,
    )

    bi_result = materialize_meeting_point_hashidxs(bi_result, puzzle, solve_config)

    assert bool(jax.device_get(bi_result.meeting.fwd_has_hashidx)) is True
    assert bool(jax.device_get(bi_result.meeting.bwd_has_hashidx)) is True

    fwd_hidx, found = bi_result.forward.hashtable.lookup(meeting_state)
    assert bool(jax.device_get(found)) is True
    assert int(jax.device_get(fwd_hidx.index)) == int(
        jax.device_get(bi_result.meeting.fwd_hashidx.index)
    )

    parent = bi_result.forward.parent[bi_result.meeting.fwd_hashidx.index]
    assert int(jax.device_get(parent.hashidx.index)) == int(jax.device_get(start_hashidx.index))
    assert int(jax.device_get(parent.action)) == action

    path = reconstruct_bidirectional_path(bi_result, puzzle)
    assert _states_equal(path[0][1], start)
    assert _states_equal(path[-1][1], goal)
    assert [a for a, _ in path] == [-1, action]
    _assert_path_replays_to_goal(puzzle, solve_config, path, goal)


def test_materialize_meeting_point_hashidxs_materializes_backward_edge_only_meeting():
    puzzle = SlidePuzzle(size=2)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)
    start, _, _ = _find_one_step_start_to_goal(puzzle, solve_config, goal)
    inverse_solveconfig = puzzle.hindsight_transform(solve_config, start)

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    (
        _,
        fwd_current,
        _,
        _,
        bwd_current,
        bwd_states,
    ) = initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    goal_hashidx = bwd_current.hashidx[0]
    assert _states_equal(jax.tree_util.tree_map(lambda x: x[0], bwd_states), goal)
    action, step_cost = _find_inverse_action_to_start(puzzle, inverse_solveconfig, goal, start)
    meeting_state = start
    meeting_fwd_hashidx = fwd_current.hashidx[0]

    dummy_hashidx = HashIdx.default(())
    dummy_action = jnp.array(0, dtype=ACTION_DTYPE)
    bi_result.meeting = MeetingPoint(
        fwd_hashidx=meeting_fwd_hashidx,
        bwd_hashidx=dummy_hashidx,
        fwd_cost=jnp.array(0, dtype=KEY_DTYPE),
        bwd_cost=step_cost,
        total_cost=step_cost,
        found=jnp.array(True),
        fwd_has_hashidx=jnp.array(True),
        bwd_has_hashidx=jnp.array(False),
        fwd_parent_hashidx=dummy_hashidx,
        fwd_parent_action=dummy_action,
        bwd_parent_hashidx=goal_hashidx,
        bwd_parent_action=jnp.array(action, dtype=ACTION_DTYPE),
    )

    bi_result = materialize_meeting_point_hashidxs(
        bi_result, puzzle, solve_config, inverse_solveconfig=inverse_solveconfig
    )

    assert bool(jax.device_get(bi_result.meeting.fwd_has_hashidx)) is True
    assert bool(jax.device_get(bi_result.meeting.bwd_has_hashidx)) is True

    bwd_hidx, found = bi_result.backward.hashtable.lookup(meeting_state)
    assert bool(jax.device_get(found)) is True
    assert int(jax.device_get(bwd_hidx.index)) == int(
        jax.device_get(bi_result.meeting.bwd_hashidx.index)
    )

    parent = bi_result.backward.parent[bi_result.meeting.bwd_hashidx.index]
    assert int(jax.device_get(parent.hashidx.index)) == int(jax.device_get(goal_hashidx.index))
    assert int(jax.device_get(parent.action)) == action

    path = reconstruct_bidirectional_path(bi_result, puzzle)
    assert _states_equal(path[0][1], start)
    assert _states_equal(path[-1][1], goal)
    assert [a for a, _ in path] == [-1, action]
    _assert_path_replays_to_goal(puzzle, solve_config, path, goal)

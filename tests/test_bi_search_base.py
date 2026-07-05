import jax
import jax.numpy as jnp
from puxle import SlidePuzzle
from xtructure import HashIdx

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE
from JAxtar.bi_stars.bi_search_base import (
    MeetingPoint,
    build_bi_search_result,
    detect_meeting,
    initialize_bi_loop_common,
    reconstruct_bidirectional_path,
    register_seen,
    update_meeting_point,
)
from JAxtar.stars.search_base import Parent


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

    # Both directions share a single hash table.
    assert result.forward.hashtable is result.backward.hashtable

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
    # start == goal collapses to a single shared slot registered by both directions.
    assert int(jax.device_get(bi_result.meeting.fwd_hashidx.index)) == int(
        jax.device_get(bi_result.meeting.bwd_hashidx.index)
    )


def test_initialize_bi_loop_common_shares_table_and_registers_roots():
    puzzle = SlidePuzzle(size=2)
    solve_config, goal = _make_valid_slide_solve_config(puzzle)
    start, _, _ = _find_one_step_start_to_goal(puzzle, solve_config, goal)

    bi_result = build_bi_search_result(
        statecls=puzzle.State,
        batch_size=8,
        max_nodes=64,
        action_size=puzzle.action_size,
    )

    _, fwd_current, _, _, bwd_current, _ = initialize_bi_loop_common(
        bi_result, puzzle, solve_config, start
    )

    # Forward and backward share one hash table object after initialization.
    assert bi_result.forward.hashtable is bi_result.backward.hashtable

    start_slot = int(jax.device_get(fwd_current.hashidx[0].index))
    goal_slot = int(jax.device_get(bwd_current.hashidx[0].index))
    assert start_slot != goal_slot

    # Each root is registered only in its own direction.
    assert bool(jax.device_get(bi_result.seen_forward[start_slot])) is True
    assert bool(jax.device_get(bi_result.seen_backward[goal_slot])) is True
    assert bool(jax.device_get(bi_result.seen_backward[start_slot])) is False
    assert bool(jax.device_get(bi_result.seen_forward[goal_slot])) is False

    # The shared table holds both roots, findable from either direction's handle.
    _, start_found = bi_result.backward.hashtable.lookup(start)
    _, goal_found = bi_result.forward.hashtable.lookup(goal)
    assert bool(jax.device_get(start_found)) is True
    assert bool(jax.device_get(goal_found)) is True


def test_meeting_via_shared_slot_reconstructs_path():
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
    goal_slot = int(jax.device_get(bwd_current.hashidx[0].index))

    # Simulate the forward frontier expanding start -> goal into the shared table.
    # The shared table already holds goal (the backward root), so forward gets the
    # SAME slot rather than a fresh one.
    bi_result.forward.hashtable, _, fwd_goal_hashidx = bi_result.forward.hashtable.insert(goal)
    assert int(jax.device_get(fwd_goal_hashidx.index)) == goal_slot

    bi_result.forward.cost = bi_result.forward.cost.at[fwd_goal_hashidx.index].set(
        jnp.asarray(step_cost, dtype=KEY_DTYPE)
    )
    bi_result.forward.parent = bi_result.forward.parent.at[
        jnp.asarray(fwd_goal_hashidx.index)[jnp.newaxis]
    ].set_as_condition(
        jnp.array([True]),
        Parent(hashidx=start_hashidx, action=jnp.array(action, dtype=ACTION_DTYPE)),
    )

    hashidxs = HashIdx(index=jnp.asarray(fwd_goal_hashidx.index)[jnp.newaxis])
    mask = jnp.array([True])
    this_costs = jnp.asarray(step_cost, dtype=KEY_DTYPE)[jnp.newaxis]

    bi_result = register_seen(bi_result, hashidxs, mask, is_forward=True)
    assert bool(jax.device_get(bi_result.seen_forward[goal_slot])) is True

    # Meeting is a pure gather against the opposite frontier's seen flags/costs.
    found_mask, opposite_costs, total_costs = detect_meeting(
        bi_result, hashidxs, mask, this_costs, is_forward=True
    )
    assert bool(jax.device_get(found_mask[0])) is True
    assert float(jax.device_get(opposite_costs[0])) == 0.0

    bi_result.meeting = update_meeting_point(
        bi_result.meeting,
        found_mask,
        hashidxs,
        this_costs,
        opposite_costs,
        total_costs,
        is_forward=True,
    )

    assert bool(jax.device_get(bi_result.meeting.found)) is True
    # A shared meeting slot: both views point at the same table entry.
    assert int(jax.device_get(bi_result.meeting.fwd_hashidx.index)) == int(
        jax.device_get(bi_result.meeting.bwd_hashidx.index)
    )
    assert float(jax.device_get(bi_result.meeting.total_cost)) == float(jax.device_get(step_cost))

    path = reconstruct_bidirectional_path(bi_result, puzzle)
    assert _states_equal(path[0][1], start)
    assert _states_equal(path[-1][1], goal)
    assert [a for a, _ in path] == [-1, action]
    _assert_path_replays_to_goal(puzzle, solve_config, path, goal)


def test_update_meeting_point_records_real_slot_on_float16_overflow():
    # A first meeting whose summed g-cost overflows KEY_DTYPE (float16) to +inf must
    # still record the real meeting slot, not flip found=True while pointing at the
    # build-time sentinel (which would corrupt stamping/reconstruction).
    dummy = HashIdx.default(())
    meeting = MeetingPoint(
        fwd_hashidx=dummy,
        bwd_hashidx=dummy,
        fwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        bwd_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        total_cost=jnp.array(jnp.inf, dtype=KEY_DTYPE),
        found=jnp.array(False),
    )

    real_slot = 5
    hashidxs = HashIdx(index=jnp.array([real_slot], dtype=dummy.index.dtype))
    found_mask = jnp.array([True])
    this_costs = jnp.array([40000.0], dtype=KEY_DTYPE)
    opposite_costs = jnp.array([40000.0], dtype=KEY_DTYPE)
    total_costs = (this_costs + opposite_costs).astype(KEY_DTYPE)
    assert bool(jax.device_get(jnp.isinf(total_costs[0]))), "expected float16 overflow"

    out = update_meeting_point(
        meeting,
        found_mask,
        hashidxs,
        this_costs,
        opposite_costs,
        total_costs,
        is_forward=True,
    )

    assert bool(jax.device_get(out.found)) is True
    assert int(jax.device_get(out.fwd_hashidx.index)) == real_slot
    assert int(jax.device_get(out.bwd_hashidx.index)) == real_slot
    assert int(jax.device_get(out.fwd_hashidx.index)) != int(jax.device_get(dummy.index))

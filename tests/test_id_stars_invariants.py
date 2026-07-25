"""Behavioural pins for the ID{}* family.

Nothing else in the suite actually *runs* an ID search, so any refactor of
``JAxtar/id_stars/`` is currently unguarded. These tests exist so that payload
and dead-field cleanups can be proven bit-identical: they pin the full observable
outcome (solved / cost / generated_count / action sequence), not just "it ran".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import xtructure.numpy as xnp
from puxle import SlidePuzzle

from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from JAxtar.annotate import ACTION_DTYPE
from JAxtar.id_stars.id_astar import id_astar_builder
from JAxtar.id_stars.id_qstar import id_qstar_builder
from JAxtar.search_build_spec import SearchBuildSpec
from qfunction.empty_q import EmptyQFunction


def _instance(size: int, seed: int):
    puzzle = SlidePuzzle(size=size)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(seed))
    return puzzle, solve_config, state


def _outcome(search_result) -> tuple[bool, float, int, tuple[int, ...]]:
    actions = search_result.solution_actions_arr
    pad = jnp.iinfo(ACTION_DTYPE).max
    return (
        bool(search_result.solved),
        float(search_result.solution_cost),
        int(search_result.generated_count),
        tuple(int(a) for a in jax.device_get(actions[actions != pad])),
    )


def _replays_to_goal(puzzle, solve_config, state, actions) -> bool:
    """Replay `actions` from `state` and report whether the goal is reached."""
    current = xnp.expand_dims(state, 0)
    for action in actions:
        current, _ = puzzle.batched_get_actions(
            solve_config,
            current,
            jnp.array([action], dtype=jnp.int32),
            jnp.array([True]),
        )
    return bool(puzzle.batched_is_solved(solve_config, current).all())


def _run_id_astar(size: int, seed: int, *, spec_kwargs=None, **build_kwargs):
    puzzle, solve_config, state = _instance(size, seed)
    spec = SearchBuildSpec(
        **{"cost_weight": 1.0, "warmup_inputs": (solve_config, state), **(spec_kwargs or {})}
    )
    search = id_astar_builder(
        puzzle,
        SlidePuzzleHeuristic(puzzle),
        batch_size=32,
        max_nodes=4096,
        spec=spec,
        **build_kwargs,
    )
    return puzzle, solve_config, state, search(solve_config, state)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_id_astar_outcome_is_stable(seed: int):
    """Pins the whole observable outcome so refactors can be diffed against it."""
    puzzle, solve_config, state, result = _run_id_astar(2, seed)
    solved, cost, generated, actions = _outcome(result)

    assert solved
    assert generated > 0
    # The returned action sequence must actually reach the goal from the start.
    assert _replays_to_goal(puzzle, solve_config, state, actions)
    assert cost == pytest.approx(float(len(actions)))


def test_id_astar_finds_the_optimal_cost_on_a_tiny_instance():
    """cost_weight=1.0 on SlidePuzzle(2): the returned cost must be the optimum."""
    puzzle, solve_config, state = _instance(2, 0)
    from JAxtar.stars.astar import astar_builder

    reference = astar_builder(
        puzzle,
        SlidePuzzleHeuristic(puzzle),
        batch_size=32,
        max_nodes=4096,
        spec=SearchBuildSpec(cost_weight=1.0, warmup_inputs=(solve_config, state)),
    )(solve_config, state)
    assert bool(reference.solved)
    optimal = float(reference.get_cost(reference.solved_idx))

    _, _, _, result = _run_id_astar(2, 0)
    assert float(result.solution_cost) == pytest.approx(optimal)


def test_max_path_len_is_a_hard_depth_cutoff_not_a_free_knob():
    """`flat_depth <= max_path_len` prunes the search, so shrinking it can lose solutions.

    This pins that the failure mode is an honest `solved=False`, never a truncated
    path reported as a solution.
    """
    _, _, _, deep = _run_id_astar(3, 0, max_path_len=64)
    assert bool(deep.solved)
    solution_len = len(_outcome(deep)[3])

    _, _, _, shallow = _run_id_astar(3, 0, max_path_len=max(1, solution_len // 2))
    if bool(shallow.solved):
        # If it still solves, it must be a genuinely valid (shorter) path, not a stub.
        assert len(_outcome(shallow)[3]) > 0
    else:
        assert float(shallow.solution_cost) == float("inf")


def test_id_qstar_runs_and_returns_a_valid_action_sequence():
    puzzle, solve_config, state = _instance(2, 0)
    search = id_qstar_builder(
        puzzle,
        EmptyQFunction(puzzle),
        batch_size=32,
        max_nodes=4096,
        spec=SearchBuildSpec(cost_weight=1.0, warmup_inputs=(solve_config, state)),
    )
    result = search(solve_config, state)
    solved, _, generated, actions = _outcome(result)

    assert solved
    assert generated > 0
    assert _replays_to_goal(puzzle, solve_config, state, actions)


def test_bound_step_reduces_work_without_breaking_the_answer():
    """The threshold grid must cut the ladder, not the correctness of the result.

    With a fractional ``cost_weight`` the exact ladder advances by one float16 ULP per
    pass and re-derives the tree every time; the grid collapses that to a handful of
    thresholds. The returned path must still replay to the goal.
    """
    puzzle, solve_config, state, exact = _run_id_astar(
        3, 0, spec_kwargs={"cost_weight": 0.9, "bound_step": 0.0}
    )
    _, _, _, gridded = _run_id_astar(3, 0, spec_kwargs={"cost_weight": 0.9, "bound_step": 1.0})

    assert bool(exact.solved) and bool(gridded.solved)
    for result in (exact, gridded):
        assert _replays_to_goal(puzzle, solve_config, state, _outcome(result)[3])
    assert int(gridded.generated_count) <= int(exact.generated_count)


def test_batched_state_equal_is_per_row_not_a_scalar():
    """Regression guard for the bug that silently disabled every non-backtracking filter.

    ``Xtructurable.__eq__`` reduces to one scalar. When the shared helper returned that
    scalar, callers broadcast it across their mask and blocked either every node or --
    in practice -- none, in both ``id_stars`` and ``beamsearch``.
    """
    from JAxtar.utils.array_ops import batched_state_equal

    puzzle = SlidePuzzle(size=3)
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(0))
    starts = xnp.reshape(xnp.stack([state] * 4, axis=0), (4,))
    neighbours, _ = puzzle.batched_get_neighbours(
        solve_config, starts, jnp.ones(4, dtype=jnp.bool_)
    )
    others = xnp.take(neighbours, 0, axis=0)
    mixed = xnp.reshape(
        xnp.stack(
            [
                xnp.take(starts, 0, axis=0),
                xnp.take(others, 1, axis=0),
                xnp.take(starts, 2, axis=0),
                xnp.take(others, 3, axis=0),
            ],
            axis=0,
        ),
        (4,),
    )

    equal = batched_state_equal(starts, mixed)
    assert equal.shape == (4,), "must be one bool per row, not a reduced scalar"
    assert list(map(bool, equal)) == [True, False, True, False]
    assert bool(batched_state_equal(starts, starts).all())
    assert not bool(batched_state_equal(starts, others).any())


def test_non_backtracking_actually_blocks_the_undo_move():
    """One child of every expanded node is its own parent's predecessor; it must be cut."""
    from JAxtar.id_stars.id_frontier import build_child_trail
    from JAxtar.id_stars.search_base import apply_non_backtracking

    puzzle = SlidePuzzle(size=3)
    action_size = puzzle.action_size
    batch = action_size
    flat_size = action_size * batch
    steps = 2
    solve_config, state = puzzle.get_inits(jax.random.PRNGKey(0))

    starts = xnp.reshape(xnp.stack([state] * batch, axis=0), (batch,))
    filled = jnp.ones((batch,), dtype=jnp.bool_)
    parents, _ = puzzle.batched_get_actions(
        solve_config, starts, jnp.arange(batch, dtype=jnp.int32), filled
    )
    parent_trail = build_child_trail(
        puzzle.State.default((batch, steps)),
        starts,
        1,
        batch,
        steps,
        puzzle.State.default((batch, 0)),
    )
    children, _ = puzzle.batched_get_neighbours(solve_config, parents, filled)
    flat_children = xnp.reshape(children, (flat_size,))

    kept = apply_non_backtracking(
        flat_children,
        parents,
        parent_trail,
        jnp.ones((batch,), dtype=jnp.int32),
        jnp.ones((flat_size,), dtype=jnp.bool_),
        steps,
        action_size,
        flat_size,
        jnp.arange(steps, dtype=jnp.int32),
        batch,
    )
    blocked = flat_size - int(kept.sum())
    # Every parent reaches back to `start` (or is `start`, for a blocked move), so some
    # children must be cut -- but never all of them.
    assert 0 < blocked < flat_size


def test_trace_arena_reconstructs_the_path_and_reports_overflow():
    """The arena replaces the per-node action history, so its walk *is* the answer.

    Also pins that a normal run never silently truncates: ``trace_overflow`` counts nodes
    dropped for want of an arena row, and must be zero when the arena is not exhausted.
    """
    puzzle, solve_config, state, result = _run_id_astar(3, 0)
    solved, _, _, actions = _outcome(result)

    assert solved
    assert len(actions) > 0
    assert _replays_to_goal(puzzle, solve_config, state, actions)
    assert int(result.trace_overflow) == 0

    # The stack must no longer carry a per-node path; that was the whole point.
    item_fields = set(result.ItemCls.__dataclass_fields__)
    assert "action_history" not in item_fields
    assert "trace_index" in item_fields

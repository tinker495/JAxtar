import jax
import jax.numpy as jnp

from JAxtar.beamsearch.search_base import (
    TRACE_INVALID,
    apply_selected_candidates,
    beam_loop_continue_if_not_solved,
    finalize_beam_search_result,
)


class _DummyPuzzle:
    def batched_is_solved(self, solve_config, states):
        return states == solve_config


class _DummyBeamResult:
    def __init__(self, depth, max_depth, beam, cost):
        self.depth = jnp.array(depth, dtype=jnp.int32)
        self.max_depth = int(max_depth)
        self.beam = jnp.array(beam, dtype=jnp.int32)
        self.beam_width = int(self.beam.shape[0])
        self.cost = jnp.array(cost, dtype=jnp.float32)
        self.dist = jnp.full_like(self.cost, jnp.inf)
        self.scores = jnp.full_like(self.cost, jnp.inf)
        self.parent_index = jnp.full((self.beam.shape[0],), -1, dtype=jnp.int32)
        self.solved = jnp.array(False)
        self.solved_idx = jnp.array(-1, dtype=jnp.int32)
        self.generated_size = jnp.array(0, dtype=jnp.int32)

    def filled_mask(self):
        return jnp.isfinite(self.cost)


def test_beam_loop_continue_if_not_solved_stops_when_solution_exists():
    puzzle = _DummyPuzzle()
    result = _DummyBeamResult(
        depth=1,
        max_depth=4,
        beam=[2, 5, 7],
        cost=[0.0, 1.0, jnp.inf],
    )

    should_continue = beam_loop_continue_if_not_solved(result, puzzle, solve_config=jnp.array(5))
    assert bool(jax.device_get(should_continue)) is False


def test_finalize_beam_search_result_sets_solution_index():
    puzzle = _DummyPuzzle()
    result = _DummyBeamResult(
        depth=1,
        max_depth=4,
        beam=[2, 9, 1],
        cost=[0.0, 1.0, jnp.inf],
    )

    finalized = finalize_beam_search_result(result, puzzle, solve_config=jnp.array(9))
    assert bool(jax.device_get(finalized.solved)) is True
    assert int(jax.device_get(finalized.solved_idx)) == 1


def test_apply_selected_candidates_updates_trace_and_masks_invalid_slots():
    result = _DummyBeamResult(
        depth=1,
        max_depth=4,
        beam=[10, 11, 12],
        cost=[1.0, 2.0, 3.0],
    )
    result.generated_size = jnp.array(5, dtype=jnp.int32)
    result.active_trace = jnp.array([3, 4, TRACE_INVALID], dtype=jnp.uint32)

    trace_capacity = (result.max_depth + 1) * result.beam.shape[0]
    result.trace_parent = jnp.full((trace_capacity,), TRACE_INVALID, dtype=jnp.uint32)
    result.trace_action = jnp.full((trace_capacity,), 255, dtype=jnp.uint8)
    result.trace_cost = jnp.full((trace_capacity,), jnp.inf, dtype=jnp.float32)
    result.trace_dist = jnp.full((trace_capacity,), jnp.inf, dtype=jnp.float32)
    result.trace_depth = jnp.full((trace_capacity,), -1, dtype=jnp.int32)
    result.trace_state = jnp.full((trace_capacity,), -1, dtype=jnp.int32)

    updated = apply_selected_candidates(
        result,
        selected_states=jnp.array([20, 21, 22], dtype=jnp.int32),
        selected_costs=jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32),
        selected_dists=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        selected_scores=jnp.array([4.0, 6.0, 8.0], dtype=jnp.float32),
        selected_actions=jnp.array([0, 1, 2], dtype=jnp.uint8),
        selected_parents=jnp.array([1, 0, 1], dtype=jnp.int32),
        selected_valid=jnp.array([True, False, True]),
    )

    assert int(jax.device_get(updated.depth)) == 2
    assert int(jax.device_get(updated.generated_size)) == 7

    active_trace = jax.device_get(updated.active_trace).tolist()
    assert active_trace == [6, int(TRACE_INVALID), 8]

    trace_parent = jax.device_get(updated.trace_parent)
    assert int(trace_parent[6]) == 4
    assert int(trace_parent[8]) == 4
    assert int(trace_parent[7]) == int(TRACE_INVALID)

    assert jax.device_get(updated.cost).tolist() == [3.0, jnp.inf, 5.0]
    assert jax.device_get(updated.parent_index).tolist() == [1, -1, 1]

import jax.numpy as jnp

import JAxtar.bi_stars.bi_qstar as bi_qstar_mod
from JAxtar.core.result import Current


class _DummyPriorityQueue:
    def __init__(self, size: int, min_key: float = jnp.inf):
        self.size = jnp.array(size, dtype=jnp.int32)
        self.key_store = jnp.array([min_key], dtype=jnp.float32)
        self.key_buffer = jnp.array([jnp.inf], dtype=jnp.float32)


class _DummyDirectionResult:
    def __init__(
        self,
        pq_size: int,
        generated_size: int = 1,
        capacity: int = 16,
        pq_min_key: float = jnp.inf,
    ):
        self.priority_queue = _DummyPriorityQueue(pq_size, min_key=pq_min_key)
        self.generated_size = jnp.array(generated_size, dtype=jnp.int32)
        self.capacity = capacity

    def get_dist(self, current):
        return jnp.zeros_like(current.cost)


class _DummyMeeting:
    def __init__(self, found: bool, total_cost: float = 0.0):
        self.found = jnp.array(found, dtype=jnp.bool_)
        self.total_cost = jnp.array(total_cost, dtype=jnp.float32)


class _DummyBiResult:
    def __init__(
        self,
        fwd_pq_size: int,
        bwd_pq_size: int,
        meeting_found: bool = False,
        meeting_total_cost: float = 0.0,
        fwd_pq_min_key: float = jnp.inf,
        bwd_pq_min_key: float = jnp.inf,
    ):
        self.forward = _DummyDirectionResult(fwd_pq_size, pq_min_key=fwd_pq_min_key)
        self.backward = _DummyDirectionResult(bwd_pq_size, pq_min_key=bwd_pq_min_key)
        self.meeting = _DummyMeeting(meeting_found, total_cost=meeting_total_cost)


class _DummyPuzzle:
    action_size = 4


class _DummyQFunction:
    def batched_q_value(self, params, states):
        del params
        return jnp.zeros((states.shape[0], 4), dtype=jnp.float32)


def test_bi_qstar_loop_body_expands_when_frontier_is_only_in_queue(monkeypatch):
    """Loop body should trigger expansion when PQ has queued nodes, even with empty current mask."""
    captured_preds = []

    def _capture_cond(pred, true_fn, false_fn, operand):
        captured_preds.append(bool(pred))
        return false_fn(operand)

    monkeypatch.setattr(bi_qstar_mod.jax.lax, "cond", _capture_cond)

    _, _, loop_body = bi_qstar_mod._bi_qstar_loop_builder(
        puzzle=_DummyPuzzle(),
        q_fn=_DummyQFunction(),
        batch_size=2,
    )

    loop_state = bi_qstar_mod.BiLoopStateWithStates(
        bi_result=_DummyBiResult(fwd_pq_size=1, bwd_pq_size=0),
        solve_config=None,
        inverse_solveconfig=None,
        params_forward=None,
        params_backward=None,
        current_forward=Current.default((2,)),
        current_backward=Current.default((2,)),
        states_forward=jnp.zeros((2,), dtype=jnp.int32),
        states_backward=jnp.zeros((2,), dtype=jnp.int32),
        filled_forward=jnp.array([False, False], dtype=jnp.bool_),
        filled_backward=jnp.array([False, False], dtype=jnp.bool_),
    )

    loop_body(loop_state)

    assert captured_preds == [True, False]


def test_common_bi_loop_condition_uses_pq_min_in_optimality_mode():
    """Optimality mode must use queued PQ lower-bound when current masks are empty."""
    should_continue = bi_qstar_mod._common_bi_loop_condition(
        bi_result=_DummyBiResult(
            fwd_pq_size=1,
            bwd_pq_size=0,
            meeting_found=True,
            meeting_total_cost=5.0,
            fwd_pq_min_key=1.0,
            bwd_pq_min_key=jnp.inf,
        ),
        filled_forward=jnp.array([False, False], dtype=jnp.bool_),
        filled_backward=jnp.array([False, False], dtype=jnp.bool_),
        current_forward=Current.default((2,)),
        current_backward=Current.default((2,)),
        cost_weight=1.0,
        terminate_on_first_solution=False,
    )

    assert bool(should_continue)

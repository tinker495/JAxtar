import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from JAxtar.core.bi_loop import BiLoopState, unified_bi_search_loop_builder
from JAxtar.core.common import loop_continue_if_not_solved
from JAxtar.core.loop import LoopState, unified_search_loop_builder
from JAxtar.core.result import Current


class _DummyPriorityQueue:
    def __init__(self, size: int, min_key: float = jnp.inf):
        self.size = jnp.array(size, dtype=jnp.int32)
        self.key_store = jnp.array([min_key], dtype=jnp.float32)
        self.key_buffer = jnp.array([jnp.inf], dtype=jnp.float32)


class _DummySearchResult:
    def __init__(self, solved: bool, pq_size: int):
        self.solved = jnp.array(solved)
        self.priority_queue = _DummyPriorityQueue(pq_size)


class _DummyPuzzle:
    action_size = 4
    State = object

    def __init__(self, solved_mask):
        self._solved_mask = jnp.array(solved_mask, dtype=jnp.bool_)

    def batched_is_solved(self, solve_config, states):
        return self._solved_mask


class _DummyExpansionPolicy:
    cost_weight = 1.0


class _DummyPriorityQueueWithScalarSize:
    def __init__(self, size: int):
        self.size = jnp.array(size, dtype=jnp.int32)


class _DummySingleSearchResult:
    def __init__(self, solved: bool, pq_size: int, generated_size: int = 1, capacity: int = 16):
        self.solved = jnp.array(solved, dtype=jnp.bool_)
        self.priority_queue = _DummyPriorityQueueWithScalarSize(pq_size)
        self.generated_size = jnp.array(generated_size, dtype=jnp.int32)
        self.capacity = capacity


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

    def get_state(self, current):
        return jnp.zeros_like(current.hashidx.index, dtype=jnp.int32)


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


def test_loop_continue_if_not_solved_keeps_popped_frontier_when_pq_drained():
    """Continue when popped frontier exists even if the priority queue is empty."""
    search_result = _DummySearchResult(solved=False, pq_size=0)
    puzzle = _DummyPuzzle([False, False])
    states = jnp.array([0, 0], dtype=jnp.int32)
    filled = jnp.array([True, False], dtype=jnp.bool_)

    should_continue = loop_continue_if_not_solved(search_result, puzzle, None, states, filled)

    assert bool(should_continue)


def test_unified_bi_loop_condition_keeps_popped_batches_when_queues_drained():
    """Bidirectional loop must continue on active popped batches even with empty queues."""
    puzzle = _DummyPuzzle([False, False])
    _, loop_condition, _ = unified_bi_search_loop_builder(
        puzzle=puzzle,
        fwd_expansion_policy=_DummyExpansionPolicy(),
        bwd_expansion_policy=_DummyExpansionPolicy(),
        batch_size=2,
        max_nodes=16,
        pop_ratio=jnp.inf,
        min_pop=1,
        pq_val_type=Current,
        terminate_on_first_solution=True,
    )

    loop_state = BiLoopState(
        bi_result=_DummyBiResult(fwd_pq_size=0, bwd_pq_size=0, meeting_found=False),
        solve_config=None,
        inverse_solve_config=None,
        heuristic_params=None,
        inverse_heuristic_params=None,
        current_fwd=Current.default((2,)),
        current_bwd=Current.default((2,)),
        filled_fwd=jnp.array([True, False], dtype=jnp.bool_),
        filled_bwd=jnp.array([True, False], dtype=jnp.bool_),
    )

    assert bool(loop_condition(loop_state))


def test_unified_bi_loop_condition_keeps_queued_work_even_if_current_is_empty():
    """Bidirectional loop must continue when PQ still has work to pop."""
    puzzle = _DummyPuzzle([False, False])
    _, loop_condition, _ = unified_bi_search_loop_builder(
        puzzle=puzzle,
        fwd_expansion_policy=_DummyExpansionPolicy(),
        bwd_expansion_policy=_DummyExpansionPolicy(),
        batch_size=2,
        max_nodes=16,
        pop_ratio=jnp.inf,
        min_pop=1,
        pq_val_type=Current,
        terminate_on_first_solution=True,
    )

    loop_state = BiLoopState(
        bi_result=_DummyBiResult(fwd_pq_size=1, bwd_pq_size=0, meeting_found=False),
        solve_config=None,
        inverse_solve_config=None,
        heuristic_params=None,
        inverse_heuristic_params=None,
        current_fwd=Current.default((2,)),
        current_bwd=Current.default((2,)),
        filled_fwd=jnp.array([False, False], dtype=jnp.bool_),
        filled_bwd=jnp.array([False, False], dtype=jnp.bool_),
    )

    assert bool(loop_condition(loop_state))


def test_unified_single_loop_condition_keeps_queue_work_when_current_is_empty():
    """Single unified loop must continue when queue has work, even if current mask is empty."""

    class _PuzzleForLoop(_DummyPuzzle):
        action_size = 4
        State = object
        SolveConfig = object

    class _NoOpExpansion(_DummyExpansionPolicy):
        def expand(self, *args, **kwargs):
            raise RuntimeError("expand should not run in loop-condition-only test")

    puzzle = _PuzzleForLoop([False, False])
    _, loop_condition, _ = unified_search_loop_builder(
        puzzle=puzzle,
        expansion_policy=_NoOpExpansion(),
        batch_size=2,
        max_nodes=16,
        pop_ratio=jnp.inf,
        min_pop=1,
        pq_val_type=Current,
    )

    loop_state = LoopState(
        search_result=_DummySingleSearchResult(
            solved=False,
            pq_size=1,
            generated_size=1,
            capacity=16,
        ),
        solve_config=None,
        heuristic_params=None,
        current=Current.default((2,)),
        filled=jnp.array([False, False], dtype=jnp.bool_),
        states=jnp.array([0, 0], dtype=jnp.int32),
    )

    assert bool(loop_condition(loop_state))


def test_unified_bi_loop_condition_optimality_uses_pq_lower_bound_when_current_empty():
    """With optimality termination, queued work must keep loop alive even if current masks are empty."""
    puzzle = _DummyPuzzle([False, False])
    _, loop_condition, _ = unified_bi_search_loop_builder(
        puzzle=puzzle,
        fwd_expansion_policy=_DummyExpansionPolicy(),
        bwd_expansion_policy=_DummyExpansionPolicy(),
        batch_size=2,
        max_nodes=16,
        pop_ratio=jnp.inf,
        min_pop=1,
        pq_val_type=Current,
        terminate_on_first_solution=False,
    )

    loop_state = BiLoopState(
        bi_result=_DummyBiResult(
            fwd_pq_size=1,
            bwd_pq_size=0,
            meeting_found=True,
            meeting_total_cost=5.0,
            fwd_pq_min_key=1.0,
            bwd_pq_min_key=jnp.inf,
        ),
        solve_config=None,
        inverse_solve_config=None,
        heuristic_params=None,
        inverse_heuristic_params=None,
        current_fwd=Current.default((2,)),
        current_bwd=Current.default((2,)),
        filled_fwd=jnp.array([False, False], dtype=jnp.bool_),
        filled_bwd=jnp.array([False, False], dtype=jnp.bool_),
    )

    assert bool(loop_condition(loop_state))


def test_unified_single_init_avoids_synthetic_start_queue_entry():
    """Unified init should not pre-seed PQ and should keep invalid lanes at inf cost."""

    class _NoOpExpansion(_DummyExpansionPolicy):
        def expand(self, *args, **kwargs):
            raise RuntimeError("expand should not run in init-only regression test")

    puzzle = SlidePuzzle(size=2)
    solve_config, start = puzzle.get_inits(jax.random.PRNGKey(0))

    init_loop_state, _, _ = unified_search_loop_builder(
        puzzle=puzzle,
        expansion_policy=_NoOpExpansion(),
        batch_size=4,
        max_nodes=64,
        pop_ratio=jnp.inf,
        min_pop=1,
        pq_val_type=Current,
    )

    loop_state = init_loop_state(solve_config, start)

    assert int(loop_state.search_result.priority_queue.size) == 0
    assert float(loop_state.current.cost[0]) == 0.0
    assert bool(jnp.all(jnp.isinf(loop_state.current.cost[1:])))

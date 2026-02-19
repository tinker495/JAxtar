import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import SlidePuzzle

import JAxtar.bi_stars.bi_qstar as bi_qstar_mod
from JAxtar.annotate import KEY_DTYPE
from JAxtar.core.common import normalize_neighbour_cost_layout
from JAxtar.core.result import Parent, ParentWithCosts, SearchResult


def test_normalize_neighbour_cost_layout_keeps_action_major_square_shape():
    neighbours = jnp.array([[10, 11], [20, 21]], dtype=jnp.int32)
    costs = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    out_neighbours, out_costs = normalize_neighbour_cost_layout(
        neighbours,
        costs,
        action_size=2,
        batch_size=2,
        layout="action_major",
    )

    assert out_neighbours.tolist() == neighbours.tolist()
    assert out_costs.tolist() == costs.tolist()


def test_normalize_neighbour_cost_layout_transposes_batch_major_shape():
    neighbours = jnp.array([[10, 20], [11, 21]], dtype=jnp.int32)
    costs = jnp.array([[1.0, 3.0], [2.0, 4.0]], dtype=jnp.float32)

    out_neighbours, out_costs = normalize_neighbour_cost_layout(
        neighbours,
        costs,
        action_size=2,
        batch_size=2,
        layout="batch_major",
    )

    assert out_neighbours.tolist() == [[10, 11], [20, 21]]
    assert out_costs.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_bi_qstar_trivial_meeting_uses_backward_hashidx_for_backward_parent():
    puzzle = SlidePuzzle(size=2)
    solve_config, _ = puzzle.get_inits(jax.random.PRNGKey(0))
    if hasattr(puzzle, "get_goal_state"):
        start = puzzle.get_goal_state(solve_config)
    else:
        start = solve_config.TargetState

    bi_result = bi_qstar_mod.build_bi_search_result(
        puzzle.State,
        batch_size=8,
        max_nodes=128,
        action_size=puzzle.action_size,
        parant_with_costs=True,
    )
    bi_qstar_mod._initialize_bi_loop_common(bi_result, puzzle, solve_config, start)

    assert bool(jnp.ravel(bi_result.meeting.found)[0])
    bwd_parent_idx = int(jnp.ravel(bi_result.meeting.bwd_parent_hashidx.index)[0])
    bwd_hash_idx = int(jnp.ravel(bi_result.meeting.bwd_hashidx.index)[0])
    assert bwd_parent_idx == bwd_hash_idx


class _BatchMajorInverseWrapper:
    def __init__(self, base_puzzle):
        self._base = base_puzzle
        self.action_size = base_puzzle.action_size
        self.State = base_puzzle.State
        self.SolveConfig = base_puzzle.SolveConfig
        self.inverse_neighbour_layout = "batch_major"

    def batched_get_inverse_neighbours(self, solve_config, states, mask):
        neighbours, costs = self._base.batched_get_inverse_neighbours(solve_config, states, mask)
        return jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), neighbours), jnp.swapaxes(
            costs, 0, 1
        )


def test_pop_full_with_actions_normalizes_batch_major_inverse_layout():
    """Deferred backward pop should normalize inverse_neighbour_layout=batch_major."""
    base_puzzle = SlidePuzzle(size=2)
    wrapped_puzzle = _BatchMajorInverseWrapper(base_puzzle)
    solve_config, start_state = base_puzzle.get_inits(jax.random.PRNGKey(0))

    search_result = SearchResult.build(
        statecls=base_puzzle.State,
        batch_size=4,
        max_nodes=128,
        action_size=base_puzzle.action_size,
        pq_val_type=ParentWithCosts,
    )
    search_result.hashtable, _, hash_idx = search_result.hashtable.insert(start_state)
    search_result.cost = search_result.cost.at[hash_idx.index].set(jnp.array(0.0, dtype=KEY_DTYPE))

    parent_hashidxs = xnp.pad(hash_idx, (0, search_result.batch_size - 1))
    parent_actions = jnp.array([1, 0, 0, 0], dtype=jnp.uint8)
    values = ParentWithCosts(
        parent=Parent(hashidx=parent_hashidxs, action=parent_actions),
        cost=jnp.zeros((search_result.batch_size,), dtype=KEY_DTYPE),
        dist=jnp.zeros((search_result.batch_size,), dtype=KEY_DTYPE),
    )
    keys = jnp.full((search_result.batch_size,), jnp.inf, dtype=KEY_DTYPE).at[0].set(1.0)
    masks = jnp.array([True, False, False, False], dtype=jnp.bool_)
    search_result = search_result.insert_batch(keys, values, masks)

    search_result, _, next_states, next_filled = search_result.pop_full_with_actions(
        puzzle=wrapped_puzzle,
        solve_config=solve_config,
        use_heuristic=False,
        is_backward=True,
    )

    expected_neighbours, _ = base_puzzle.batched_get_inverse_neighbours(
        solve_config, xnp.pad(start_state, (0, search_result.batch_size - 1)), masks
    )
    expected_state = expected_neighbours[parent_actions[0], 0]
    actual_state = next_states[0]

    assert bool(next_filled[0])
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual_state),
        jax.tree_util.tree_leaves(expected_state),
        strict=True,
    ):
        assert bool(jnp.array_equal(actual_leaf, expected_leaf))

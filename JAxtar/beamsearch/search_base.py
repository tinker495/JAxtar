"""Beam-search utilities that operate without hash tables or heaps.

The data structures here keep only the active beam at each depth. This keeps
memory usage predictable and small, which is essential for accelerator-bound
workloads. Selection is handled via simple top-k filtering on the candidate
scores, making the module reusable by both heuristic- and Q-based beam search
builders.
"""

import math
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle
from xtructure import Xtructurable, base_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE

ACTION_PAD = jnp.array(jnp.iinfo(ACTION_DTYPE).max, dtype=ACTION_DTYPE)
TRACE_INDEX_DTYPE = jnp.uint32
TRACE_INVALID_INT = int(jnp.iinfo(TRACE_INDEX_DTYPE).max)
TRACE_INVALID = jnp.array(TRACE_INVALID_INT, dtype=TRACE_INDEX_DTYPE)


@base_dataclass
class BeamSearchResult:
    """Compact container for the active beam."""

    cost: chex.Array
    dist: chex.Array
    scores: chex.Array
    depth: chex.Array
    parent_index: chex.Array
    beam: Xtructurable
    solved: chex.Array
    solved_idx: chex.Array
    generated_size: chex.Array
    trace_parent: chex.Array
    trace_action: chex.Array
    trace_cost: chex.Array
    trace_dist: chex.Array
    trace_depth: chex.Array
    trace_state: Xtructurable
    active_trace: chex.Array

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
    def build(
        statecls: Puzzle.State,
        beam_width: int,
        max_depth: int,
    ) -> "BeamSearchResult":
        """Allocate fixed-size buffers for the beam."""
        cost = jnp.full((beam_width,), jnp.inf, dtype=KEY_DTYPE)
        dist = jnp.full((beam_width,), jnp.inf, dtype=KEY_DTYPE)
        scores = jnp.full((beam_width,), jnp.inf, dtype=KEY_DTYPE)
        depth = jnp.array(0, dtype=jnp.int32)
        parent_index = jnp.full((beam_width,), -1, dtype=jnp.int32)
        beam = statecls.default((beam_width,))
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)
        generated_size = jnp.array(1, dtype=jnp.int32)

        trace_capacity = (max_depth + 1) * beam_width
        trace_parent = jnp.full((trace_capacity,), TRACE_INVALID, dtype=TRACE_INDEX_DTYPE)
        trace_action = jnp.full((trace_capacity,), ACTION_PAD, dtype=ACTION_DTYPE)
        trace_cost = jnp.full((trace_capacity,), jnp.inf, dtype=KEY_DTYPE)
        trace_dist = jnp.full((trace_capacity,), jnp.inf, dtype=KEY_DTYPE)
        trace_depth = jnp.full((trace_capacity,), -1, dtype=jnp.int32)
        active_trace = jnp.full((beam_width,), TRACE_INVALID, dtype=TRACE_INDEX_DTYPE)
        trace_state = statecls.default((trace_capacity,))

        return BeamSearchResult(
            cost=cost,
            dist=dist,
            scores=scores,
            depth=depth,
            parent_index=parent_index,
            beam=beam,
            solved=solved,
            solved_idx=solved_idx,
            generated_size=generated_size,
            trace_parent=trace_parent,
            trace_action=trace_action,
            trace_cost=trace_cost,
            trace_dist=trace_dist,
            trace_depth=trace_depth,
            trace_state=trace_state,
            active_trace=active_trace,
        )

    def filled_mask(self) -> chex.Array:
        return jnp.isfinite(self.cost)

    def solution_actions(self) -> list[int]:
        """Return the action sequence that reaches the solved slot."""
        trace = self._reconstruct_trace()
        if trace is None:
            return []
        _, _, actions = trace
        return actions

    def get_state(self, idx: int) -> Puzzle.State:
        """Return the state at `idx` in the current beam."""
        return self.beam[int(idx)]

    def get_cost(self, idx: int) -> chex.Array:
        """Return the accumulated path cost for the slot."""
        return self.cost[int(idx)]

    def get_dist(self, idx: int) -> chex.Array:
        """Return the cached heuristic/Q distance for the slot."""
        return self.dist[int(idx)]

    def get_solved_path(self):
        """Return the sequence of states along the solved path."""
        return []

    def solution_trace(self):
        """Return (states, costs, dists, actions) describing the solved path."""
        trace = self._reconstruct_trace()
        if trace is None:
            return [], [], [], []
        costs, dists, actions = trace
        return [], costs, dists, actions

    def _reconstruct_trace(self):
        solved_idx = int(self.solved_idx)
        if solved_idx < 0:
            return None

        active_trace = np.asarray(jax.device_get(self.active_trace), dtype=np.uint32)
        node_idx = int(active_trace[solved_idx])
        if node_idx == TRACE_INVALID_INT:
            return None

        trace_parent = np.asarray(jax.device_get(self.trace_parent), dtype=np.uint32)
        trace_action = jax.device_get(self.trace_action)
        trace_cost = jax.device_get(self.trace_cost)
        trace_dist = jax.device_get(self.trace_dist)

        costs_rev = []
        dists_rev = []
        actions_rev = []

        invalid = TRACE_INVALID_INT

        while node_idx != invalid:
            cost_val = float(trace_cost[node_idx])
            dist_raw = float(trace_dist[node_idx])
            dists_rev.append(dist_raw if math.isfinite(dist_raw) else None)
            costs_rev.append(cost_val)

            parent_idx = int(trace_parent[node_idx])
            if parent_idx != invalid:
                actions_rev.append(int(trace_action[node_idx]))
            node_idx = parent_idx

        costs = list(reversed(costs_rev))
        dists = list(reversed(dists_rev))
        actions = list(reversed(actions_rev))
        return costs, dists, actions


def select_beam(
    scores: chex.Array,
    beam_width: int,
    pop_ratio: float = jnp.inf,
    min_keep: int = 1,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Select the top-scoring candidates for the next beam.

    Args:
        scores: Flattened candidate scores with shape (N,) where N is the number
            of generated children in the current expansion step. Invalid
            candidates should already be encoded as `jnp.inf`.
        beam_width: Maximum number of elements that survive the pruning step.
        pop_ratio: Multiplicative slack around the best score. Candidates above
            `best * pop_ratio` are dropped unless needed to satisfy `min_keep`.
        min_keep: Guarantees that at least this many valid candidates survive,
            provided they exist.

    Returns:
        Tuple of `(selected_scores, selected_indices, keep_mask)`:
            - selected_scores: Scores of the chosen candidates, shape
              `(beam_width,)`, padded with `inf` when fewer candidates exist.
            - selected_indices: Indices into the flattened candidate arrays that
              identify which entries were selected for the beam.
            - keep_mask: Boolean mask of shape `(beam_width,)` denoting which
              selected slots are valid.
    """
    chex.assert_rank(scores, 1)
    safe_scores = jnp.where(jnp.isfinite(scores), scores, jnp.inf)
    neg_scores = -safe_scores
    _, topk_idx = jax.lax.top_k(neg_scores, beam_width)
    selected_scores = safe_scores[topk_idx]
    selected_valid = jnp.isfinite(selected_scores)

    best_score = selected_scores[0]
    best_valid = selected_valid[0]
    threshold = jnp.where(best_valid, best_score * pop_ratio + 1e-6, jnp.inf)
    within_ratio = jnp.logical_and(selected_valid, jnp.less_equal(selected_scores, threshold))

    valid_count = jnp.sum(selected_valid)
    forced_keep = jnp.arange(beam_width) < jnp.minimum(min_keep, valid_count)

    keep_mask = jnp.logical_or(within_ratio, forced_keep)
    keep_mask = jnp.logical_and(keep_mask, selected_valid)

    return selected_scores, topk_idx, keep_mask


def _leafwise_all_equal(lhs: chex.Array, rhs: chex.Array) -> chex.Array:
    eq = jnp.equal(lhs, rhs)
    if eq.ndim <= 1:
        return eq
    axes = tuple(range(1, eq.ndim))
    return jnp.all(eq, axis=axes)


def _batched_state_equal(lhs: Xtructurable, rhs: Xtructurable) -> chex.Array:
    equality_tree = lhs == rhs
    leaves, _ = jax.tree_util.tree_flatten(equality_tree)
    if not leaves:
        raise ValueError("State comparison received an empty tree")
    result = leaves[0]
    for leaf in leaves[1:]:
        result = jnp.logical_and(result, leaf)
    return result


def non_backtracking_mask(
    candidate_states: Xtructurable,
    parent_trace_ids: chex.Array,
    trace_state: Xtructurable,
    trace_parent: chex.Array,
    lookback: int,
) -> chex.Array:
    if lookback <= 0:
        return jnp.ones(parent_trace_ids.shape, dtype=jnp.bool_)

    def _scan_fn(carry, _):
        trace_ids, blocked = carry
        valid = trace_ids != TRACE_INVALID
        safe_ids = jnp.where(valid, trace_ids, jnp.zeros_like(trace_ids))
        safe_ids_int = safe_ids.astype(jnp.int32)

        ancestor_states = trace_state[safe_ids_int]
        matches = _batched_state_equal(candidate_states, ancestor_states)
        matches = jnp.logical_and(matches, valid)
        blocked = jnp.logical_or(blocked, matches)

        parent_ids = trace_parent[safe_ids_int]
        parent_ids = jnp.where(valid, parent_ids, TRACE_INVALID)
        return (parent_ids, blocked), None

    init_blocked = jnp.zeros(parent_trace_ids.shape, dtype=jnp.bool_)
    (_, blocked), _ = jax.lax.scan(
        _scan_fn,
        (parent_trace_ids, init_blocked),
        xs=None,
        length=lookback,
    )
    return jnp.logical_not(blocked)


__all__ = [
    "BeamSearchResult",
    "select_beam",
    "TRACE_INVALID",
    "TRACE_INVALID_INT",
    "TRACE_INDEX_DTYPE",
    "non_backtracking_mask",
]

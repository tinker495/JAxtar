"""Beam-search utilities that operate without hash tables or heaps.

The data structures here keep only the active beam at each depth. This keeps
memory usage predictable and small, which is essential for accelerator-bound
workloads. Selection is handled via simple top-k filtering on the candidate
scores, making the module reusable by both heuristic- and Q-based beam search
builders.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import Xtructurable, base_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE

ACTION_PAD = jnp.array(jnp.iinfo(ACTION_DTYPE).max, dtype=ACTION_DTYPE)


@base_dataclass
class BeamSearchResult:
    """Compact container for the active beam."""

    cost: chex.Array
    dist: chex.Array
    scores: chex.Array
    depth: chex.Array
    path_actions: chex.Array
    parent_index: chex.Array
    path_states: Xtructurable
    path_costs: chex.Array
    path_dists: chex.Array
    beam: Xtructurable
    solved: chex.Array
    solved_idx: chex.Array
    generated_size: chex.Array

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
        path_actions = jnp.full((beam_width, max_depth), ACTION_PAD, dtype=ACTION_DTYPE)
        parent_index = jnp.full((beam_width,), -1, dtype=jnp.int32)
        path_states = statecls.default((beam_width, max_depth + 1))
        path_costs = jnp.full((beam_width, max_depth + 1), jnp.inf, dtype=KEY_DTYPE)
        path_dists = jnp.full((beam_width, max_depth + 1), jnp.inf, dtype=KEY_DTYPE)
        beam = statecls.default((beam_width,))
        solved = jnp.array(False)
        solved_idx = jnp.array(-1, dtype=jnp.int32)
        generated_size = jnp.array(1, dtype=jnp.int32)

        return BeamSearchResult(
            cost=cost,
            dist=dist,
            scores=scores,
            depth=depth,
            path_actions=path_actions,
            parent_index=parent_index,
            path_states=path_states,
            path_costs=path_costs,
            path_dists=path_dists,
            beam=beam,
            solved=solved,
            solved_idx=solved_idx,
            generated_size=generated_size,
        )

    def filled_mask(self) -> chex.Array:
        return jnp.isfinite(self.cost)

    def solution_actions(self) -> list[int]:
        """Return the action sequence that reaches the solved slot."""
        solved_idx = int(self.solved_idx)
        if solved_idx < 0:
            return []
        actions = self.path_actions[solved_idx]
        valid = actions != ACTION_PAD
        return [int(a) for a in actions[valid].tolist()]

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
        solved_idx = int(self.solved_idx)
        if solved_idx < 0:
            return []
        actions = self.path_actions[solved_idx]
        valid = actions != ACTION_PAD
        path_len = int(jnp.sum(valid))
        states_row = self.path_states[solved_idx]
        return [states_row[i] for i in range(path_len + 1)]

    def solution_trace(self):
        """Return (states, costs, dists, actions) describing the solved path."""
        solved_idx = int(self.solved_idx)
        if solved_idx < 0:
            return [], [], [], []
        actions = self.path_actions[solved_idx]
        valid = actions != ACTION_PAD
        path_len = int(jnp.sum(valid))
        states_row = self.path_states[solved_idx]
        costs_row = self.path_costs[solved_idx]
        dists_row = self.path_dists[solved_idx]
        states = [states_row[i] for i in range(path_len + 1)]
        costs = [float(costs_row[i]) for i in range(path_len + 1)]
        dists = []
        for i in range(path_len + 1):
            val = float(dists_row[i])
            dists.append(val if bool(jnp.isfinite(dists_row[i])) else None)
        valid_actions = actions[valid]
        actions_list = [int(a) for a in valid_actions.tolist()]
        return states, costs, dists, actions_list


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
    sorted_idx = jnp.argsort(safe_scores)
    topk_idx = sorted_idx[:beam_width]
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


__all__ = ["BeamSearchResult", "select_beam"]

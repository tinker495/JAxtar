import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle


def _batched_state_equal(lhs: Puzzle.State, rhs: Puzzle.State) -> jnp.ndarray:
    """
    Checks if two batches of states are equal element-wise.
    """
    equality_tree = lhs == rhs
    leaves, _ = jax.tree_util.tree_flatten(equality_tree)
    if not leaves:
        raise ValueError("State comparison received an empty tree")
    result = leaves[0]
    for leaf in leaves[1:]:
        result = jnp.logical_and(result, leaf)
    return result


def _apply_non_backtracking(
    candidate_states: Puzzle.State,
    parent_states: Puzzle.State,
    parent_trail: Puzzle.State,
    parent_depths: jnp.ndarray,
    valid_mask: jnp.ndarray,
    non_backtracking_steps: int,
    action_size: int,
    flat_size: int,
    trail_indices: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """
    Applies non-backtracking pruning to a batch of candidate states.
    Prevents moving back to the parent or any state in the recent trail.
    """
    if non_backtracking_steps <= 0:
        return valid_mask

    parent_states_tiled = xnp.stack([parent_states] * action_size, axis=0)
    flat_parent_states = xnp.reshape(parent_states_tiled, (flat_size,))
    blocked = _batched_state_equal(candidate_states, flat_parent_states)
    blocked = jnp.logical_and(blocked, valid_mask)

    parent_trail_tiled = xnp.stack([parent_trail] * action_size, axis=0)
    flat_trail = xnp.reshape(parent_trail_tiled, (flat_size, non_backtracking_steps))
    flat_parent_depths = jnp.broadcast_to(parent_depths, (action_size, batch_size)).reshape(
        (flat_size,)
    )

    def _loop(i, carry):
        trail_state = xnp.take(flat_trail, trail_indices[i], axis=1)
        matches = _batched_state_equal(candidate_states, trail_state)
        valid_trail = trail_indices[i] < flat_parent_depths
        matches = jnp.logical_and(matches, valid_trail)
        return jnp.logical_or(carry, matches)

    blocked = jax.lax.fori_loop(0, non_backtracking_steps, _loop, blocked)
    return jnp.logical_and(valid_mask, jnp.logical_not(blocked))

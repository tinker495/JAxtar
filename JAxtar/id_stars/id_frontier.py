from typing import Any

import chex
import jax
import jax.numpy as jnp
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass

from JAxtar.annotate import ACTION_DTYPE, KEY_DTYPE

ACTION_PAD = jnp.array(jnp.iinfo(ACTION_DTYPE).max, dtype=ACTION_DTYPE)


def validate_non_backtracking_steps(steps: int) -> int:
    """
    Validate and convert non_backtracking_steps parameter.

    Args:
        steps: Number of non-backtracking steps to validate

    Returns:
        Validated steps as integer

    Raises:
        ValueError: If steps is negative
    """
    if steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    return int(steps)


@base_dataclass
class IDFrontier:
    states: Xtructurable
    costs: chex.Array
    depths: chex.Array
    valid_mask: chex.Array
    f_scores: chex.Array
    trail: Xtructurable
    action_history: chex.Array

    solved: chex.Array
    solution_state: Xtructurable
    solution_cost: chex.Array
    solution_actions_arr: Xtructurable

    @staticmethod
    def initialize_from_start(
        puzzle: Puzzle,
        solve_config: Puzzle.SolveConfig,
        start: Puzzle.State,
        batch_size: int,
        non_backtracking_steps: int,
        max_path_len: int,
    ) -> "IDFrontier":
        start_reshaped = xnp.expand_dims(start, axis=0)
        root_solved = puzzle.batched_is_solved(solve_config, start_reshaped)[0]

        start_padded = xnp.pad(start_reshaped, ((0, batch_size - 1),), mode="constant")
        trail_padded = puzzle.State.default((batch_size, non_backtracking_steps))

        costs_padded = jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE)
        costs_padded = costs_padded.at[0].set(0.0)
        depths_padded = jnp.full((batch_size,), 0, dtype=jnp.int32)
        valid_padded = jnp.zeros((batch_size,), dtype=jnp.bool_)
        valid_padded = valid_padded.at[0].set(True)

        action_history_padded = jnp.full((batch_size, max_path_len), ACTION_PAD, dtype=ACTION_DTYPE)

        solution_state = start_reshaped
        solution_cost = jax.lax.cond(
            root_solved,
            lambda _: jnp.array(0, dtype=KEY_DTYPE),
            lambda _: jnp.array(jnp.inf, dtype=KEY_DTYPE),
            None,
        )
        solution_actions = jnp.full((max_path_len,), ACTION_PAD, dtype=ACTION_DTYPE)

        return IDFrontier(
            states=start_padded,
            costs=costs_padded,
            depths=depths_padded,
            valid_mask=valid_padded,
            f_scores=costs_padded,
            trail=trail_padded,
            solved=root_solved,
            solution_state=solution_state,
            solution_cost=solution_cost,
            solution_actions_arr=solution_actions,
            action_history=action_history_padded,
        )

    def select_top_k(
        self,
        flat_batch: Any,
        flat_valid: chex.Array,
        f_safe: chex.Array,
        batch_size: int,
        new_solved: chex.Array,
        new_sol_state: Xtructurable,
        new_sol_cost: chex.Array,
        new_sol_actions: chex.Array,
    ) -> "IDFrontier":
        flat_size = flat_valid.shape[0]
        valid_count = jnp.sum(flat_valid.astype(jnp.int32))

        def _empty_frontier(_):
            # Keep existing storage layout but mark all entries invalid.
            return IDFrontier(
                states=self.states,
                costs=self.costs,
                depths=self.depths,
                valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
                f_scores=jnp.full((batch_size,), jnp.inf, dtype=KEY_DTYPE),
                trail=self.trail,
                solved=new_solved,
                solution_state=new_sol_state,
                solution_cost=new_sol_cost,
                solution_actions_arr=new_sol_actions,
                action_history=self.action_history,
            )

        def _select_from_candidates(_):
            packed_batch, packed_valid, _, packed_idx = compact_by_valid(flat_batch, flat_valid)
            packed_f = jax.lax.cond(
                valid_count == flat_size,
                lambda __: f_safe,
                lambda __: jnp.where(packed_valid, f_safe[packed_idx], jnp.inf),
                operand=None,
            )

            neg_f = -packed_f
            top_vals, top_indices = jax.lax.top_k(neg_f, batch_size)
            selected_f = (-top_vals).astype(KEY_DTYPE)
            selected_valid = jnp.isfinite(selected_f)

            selected = xnp.take(packed_batch, top_indices, axis=0)
            selected_valid = jax.lax.cond(
                valid_count == flat_size,
                lambda __: selected_valid,
                lambda __: jnp.logical_and(selected_valid, packed_valid[top_indices]),
                operand=None,
            )

            return IDFrontier(
                states=selected.state,
                costs=selected.cost,
                depths=selected.depth,
                valid_mask=selected_valid,
                f_scores=selected_f,
                trail=selected.trail,
                solved=new_solved,
                solution_state=new_sol_state,
                solution_cost=new_sol_cost,
                solution_actions_arr=new_sol_actions,
                action_history=selected.action_history,
            )

        return jax.lax.cond(
            valid_count > 0,
            _select_from_candidates,
            _empty_frontier,
            operand=None,
        )


def compact_by_valid(
    values: Any,
    valid_mask: chex.Array,
):
    flat_size = valid_mask.shape[0]
    row_indices = jnp.arange(flat_size, dtype=jnp.int32)
    valid_count = jnp.sum(valid_mask.astype(jnp.int32))

    def _dense(_):
        packed_valid = jnp.ones((flat_size,), dtype=jnp.bool_)
        return values, packed_valid, valid_count, row_indices

    def _sparse(_):
        valid_idx = jnp.nonzero(valid_mask, size=flat_size, fill_value=0)[0].astype(jnp.int32)
        packed_values = xnp.take(values, valid_idx, axis=0)
        packed_valid = row_indices < valid_count
        return packed_values, packed_valid, valid_count, valid_idx

    def _empty(_):
        packed_valid = jnp.zeros((flat_size,), dtype=jnp.bool_)
        zero_idx = jnp.zeros((flat_size,), dtype=jnp.int32)
        return values, packed_valid, valid_count, zero_idx

    return jax.lax.cond(
        valid_count > 0,
        lambda _: jax.lax.cond(valid_count == flat_size, _dense, _sparse, operand=None),
        _empty,
        operand=None,
    )


def build_id_node_batch(statecls, non_backtracking_steps: int, max_path_len: int):
    trail_shape = (int(non_backtracking_steps),) if non_backtracking_steps > 0 else (0,)

    @xtructure_dataclass
    class IDNodeBatch:
        state: FieldDescriptor.scalar(dtype=statecls)
        cost: FieldDescriptor.scalar(dtype=KEY_DTYPE)
        depth: FieldDescriptor.scalar(dtype=jnp.int32)
        trail: FieldDescriptor.tensor(dtype=statecls, shape=trail_shape)
        action_history: FieldDescriptor.tensor(dtype=ACTION_DTYPE, shape=(max_path_len,))
        action: FieldDescriptor.scalar(dtype=jnp.int32)
        parent_index: FieldDescriptor.scalar(dtype=jnp.int32)
        root_index: FieldDescriptor.scalar(dtype=jnp.int32)

    return IDNodeBatch


def build_child_trail(
    trail: Xtructurable,
    states: Xtructurable,
    action_size: int,
    flat_size: int,
    non_backtracking_steps: int,
    empty_trail_flat: Xtructurable,
) -> Xtructurable:
    if non_backtracking_steps > 0:
        parent_trail_tiled = xnp.stack([trail] * action_size, axis=0)
        parent_states_tiled = xnp.stack([states] * action_size, axis=0)
        parent_state_exp = xnp.expand_dims(parent_states_tiled, axis=2)
        shifted_trail = parent_trail_tiled[:, :, :-1]
        child_trail = xnp.concatenate((parent_state_exp, shifted_trail), axis=2)
        return xnp.reshape(child_trail, (flat_size, non_backtracking_steps))
    return empty_trail_flat


def update_action_history(
    action_history: chex.Array,
    depth: chex.Array,
    action_ids: chex.Array,
    action_size: int,
    batch_size: int,
    flat_size: int,
    max_path_len: int,
):
    flat_action_history = jnp.broadcast_to(
        action_history[None, :, :], (action_size, batch_size, max_path_len)
    )
    flat_action_history = flat_action_history.reshape((flat_size, max_path_len))
    flat_actions = jnp.repeat(action_ids, batch_size)

    flat_depth_int = depth.astype(jnp.int32)
    flat_depth_tiled = jnp.broadcast_to(flat_depth_int[None, :], (action_size, batch_size))
    flat_depth_flat = flat_depth_tiled.reshape((flat_size,))

    safe_depth = jnp.minimum(flat_depth_flat, max_path_len - 1)
    flat_action_history = flat_action_history.at[jnp.arange(flat_size), safe_depth].set(
        flat_actions.astype(ACTION_DTYPE)
    )

    return flat_action_history, flat_actions, flat_depth_flat


def build_flat_children(
    neighbours: Xtructurable,
    step_costs: chex.Array,
    parent_costs: chex.Array,
    parent_depths: chex.Array,
    parent_states: Xtructurable,
    parent_trails: Xtructurable,
    parent_action_history: chex.Array,
    action_ids: chex.Array,
    action_size: int,
    batch_size: int,
    flat_size: int,
    non_backtracking_steps: int,
    max_path_len: int,
    empty_trail_flat: Xtructurable,
    parent_valid: chex.Array,
):
    child_g = (parent_costs[jnp.newaxis, :] + step_costs).astype(KEY_DTYPE)
    child_depth = parent_depths + 1

    flat_trail = build_child_trail(
        parent_trails,
        parent_states,
        action_size,
        flat_size,
        non_backtracking_steps,
        empty_trail_flat,
    )

    flat_action_history, flat_actions, flat_depth_flat = update_action_history(
        parent_action_history,
        parent_depths,
        action_ids,
        action_size,
        batch_size,
        flat_size,
        max_path_len,
    )

    flat_states = xnp.reshape(neighbours, (flat_size,))
    flat_g = child_g.reshape((flat_size,))
    flat_depth = jnp.broadcast_to(child_depth, (action_size, batch_size)).reshape((flat_size,))

    flat_parent_valid = jnp.broadcast_to(parent_valid, (action_size, batch_size)).reshape(
        (flat_size,)
    )
    flat_valid = jnp.logical_and(flat_parent_valid, jnp.isfinite(flat_g))

    flat_action_history = jnp.where(
        flat_valid[:, None],
        flat_action_history,
        jnp.full_like(flat_action_history, ACTION_PAD),
    )

    return (
        flat_states,
        flat_g,
        flat_depth,
        flat_trail,
        flat_action_history,
        flat_actions,
        flat_valid,
    )

import math
from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
import xtructure.numpy as xnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from qfunction.neuralq.neuralq_base import QModelBase
from train_util.losses import loss_from_diff
from train_util.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
)


def qlearning_builder(
    minibatch_size: int,
    q_fn: QModelBase,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: Optional[dict[str, Any]] = None,
    replay_ratio: int = 1,
    td_error_clip: Optional[float] = None,
):
    def qlearning_loss(
        q_params: Any,
        solveconfigs: chex.Array,
        states: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        weights: chex.Array,
    ):
        # Preprocess during training
        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        q_values, variable_updates = apply_with_conditional_batch_stats(
            q_fn.apply, q_params, preproc, training=True, n_devices=n_devices
        )
        new_params = build_new_params_from_updates(q_params, variable_updates)
        q_values_at_actions = jnp.take_along_axis(
            q_values, actions, axis=1
        )  # [batch_size, minibatch_size, 1 or 2]
        diff = target_qs - q_values_at_actions  # [batch_size, minibatch_size, 1 or 2]
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        per_sample = loss_from_diff(diff, loss=loss_type, loss_args=loss_args)
        loss_value = jnp.mean(per_sample * weights[:, jnp.newaxis])
        current_q = q_values_at_actions
        return loss_value, (new_params, diff, current_q)

    def qlearning(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: Any,
        opt_state: optax.OptState,
    ):
        """Run one optimization epoch of neural Q-learning for the provided puzzle dataset."""
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            solveconfigs, states, target_q, actions, weights = batched_dataset
            (loss, (q_params, diff, current_q)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                q_params,
                solveconfigs,
                states,
                actions,
                target_q,
                weights,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=q_params)
            q_params = optax.apply_updates(q_params, updates)
            # Calculate gradient magnitude mean
            grad_magnitude = jax.tree_util.tree_map(
                lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(grads["params"])
            )
            grad_magnitude_mean = jnp.mean(jnp.concatenate(grad_magnitude))
            return (q_params, opt_state), (loss, grad_magnitude_mean, diff, current_q)

        def replay_loop(carry, replay_key):
            q_params, opt_state = carry

            key_perm, key_fill = jax.random.split(replay_key)
            batch_indexs = jnp.concatenate(
                [
                    jax.random.permutation(key_perm, jnp.arange(data_size)),
                    jax.random.randint(
                        key_fill,
                        (batch_size * minibatch_size - data_size,),
                        0,
                        data_size,
                    ),
                ],
                axis=0,
            )
            batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

            batched_solveconfigs = xnp.take(solveconfigs, batch_indexs, axis=0)
            batched_states = xnp.take(states, batch_indexs, axis=0)
            batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
            batched_actions = jnp.take(actions, batch_indexs, axis=0)
            batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)
            batched_weights = batched_weights / (
                jnp.mean(batched_weights, axis=1, keepdims=True) + 1e-8
            )

            (q_params, opt_state,), (
                losses,
                grad_magnitude_means,
                diffs,
                current_qs,
            ) = jax.lax.scan(
                train_loop,
                (q_params, opt_state),
                (
                    batched_solveconfigs,
                    batched_states,
                    batched_target_q,
                    batched_actions,
                    batched_weights,
                ),
            )
            return (q_params, opt_state), (losses, grad_magnitude_means, diffs, current_qs)

        replay_keys = jax.random.split(key, replay_ratio)
        (q_params, opt_state,), (losses, grad_magnitude_means, diffs, current_qs,) = jax.lax.scan(
            replay_loop,
            (q_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        diffs = diffs.reshape(-1)
        current_qs = current_qs.reshape(-1)
        # Calculate weights magnitude means
        grad_magnitude_mean = jnp.mean(grad_magnitude_means)
        weights_magnitude = jax.tree_util.tree_map(
            lambda x: jnp.abs(jnp.reshape(x, (-1,))), jax.tree_util.tree_leaves(q_params["params"])
        )
        weights_magnitude_mean = jnp.mean(jnp.concatenate(weights_magnitude))
        return (
            q_params,
            opt_state,
            loss,
            grad_magnitude_mean,
            weights_magnitude_mean,
            diffs,
            current_qs,
        )

    if n_devices > 1:

        def pmap_qlearning(key, dataset, q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (
                qfunc_params,
                opt_state,
                loss,
                grad_magnitude,
                weight_magnitude,
                diffs,
                current_qs,
            ) = jax.pmap(qlearning, in_axes=(0, 0, None, None), axis_name="devices")(
                keys, dataset, q_params, opt_state
            )
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            diffs = diffs.reshape(-1)
            current_qs = current_qs.reshape(-1)
            return (
                qfunc_params,
                opt_state,
                loss,
                grad_magnitude,
                weight_magnitude,
                diffs,
                current_qs,
            )

        return pmap_qlearning
    else:
        return jax.jit(qlearning)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
) -> chex.Array:
    # Determine valid entries before sanitizing infinities
    mask = jnp.isfinite(q_values)
    q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)

    # Scale Q-values by temperature for softmax
    safe_temperature = jnp.maximum(temperature, 1e-8)
    scaled_q_values = -q_values / safe_temperature

    # Apply mask before softmax to avoid overflow
    masked_q_values = jnp.where(mask, scaled_q_values, -jnp.inf)
    probs = jax.nn.softmax(masked_q_values, axis=1)
    probs = jnp.where(mask, probs, 0.0)

    # Row-wise normalization with guard
    row_sum = jnp.sum(probs, axis=1, keepdims=True)
    probs = jnp.where(row_sum > 0.0, probs / row_sum, probs)

    # Calculate uniform probabilities
    valid_actions = jnp.sum(mask, axis=1, keepdims=True)
    uniform_valid = jnp.where(mask, 1.0 / jnp.maximum(valid_actions, 1.0), 0.0)

    action_size = q_values.shape[1]
    uniform_all = jnp.ones_like(probs) / jnp.maximum(action_size, 1)

    # Fallback if no valid actions in a row
    probs = jnp.where(valid_actions > 0, probs, uniform_all)

    # ε-greedy mixing and final guard renormalization
    probs = probs * (1.0 - epsilon) + uniform_valid * epsilon
    probs = probs / (jnp.sum(probs, axis=1, keepdims=True) + 1e-8)
    return probs


def _get_datasets_with_policy(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: QModelBase,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_double_dqn: bool = False,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states = vals
        # Check if the current states are already in a solved configuration.
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        # Preprocess the states to be suitable for neural network input.
        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        # Get the Q-values Q(s,a) for all actions 'a' in the current state 's' using the online Q-network.
        q_values = q_model.apply(q_params, preproc, training=False)
        q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)
        # Get all possible neighbor states (s') and the costs c(s,a,s') to move to them.
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        cost = jnp.transpose(cost, (1, 0))
        valid_action_mask = jnp.isfinite(cost)
        has_valid_action = jnp.any(valid_action_mask, axis=1)
        # q_sum_cost = q_values + cost
        q_sum_cost = jnp.where(jnp.isfinite(cost), q_values, jnp.inf)

        # Select an action 'a' probabilistically using a Boltzmann (softmax) exploration policy.
        # Actions with lower Q-values (lower cost-to-go) are more likely to be chosen.
        # Epsilon-greedy exploration is also mixed in.
        probs = boltzmann_action_selection(q_sum_cost, temperature=temperature)
        probs = jnp.where(valid_action_mask, probs, 0.0)
        probs_sum = jnp.sum(probs, axis=1, keepdims=True)
        probs = jnp.where(probs_sum > 0.0, probs / (probs_sum + 1e-8), probs)
        uniform_all = jnp.ones_like(probs) / jnp.maximum(probs.shape[1], 1)
        probs = jnp.where(has_valid_action[:, jnp.newaxis], probs, uniform_all)
        # Action entropy per state (measure of policy sharpness)
        entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, a_min=1e-12)), axis=1)
        # Maximum entropy per state (approx by number of valid actions)
        action_size = q_values.shape[1]
        max_ent_val = jnp.log(jnp.maximum(jnp.array(action_size, dtype=probs.dtype), 1.0))
        max_entropy = jnp.full((probs.shape[0],), max_ent_val)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )
        # Get the Q-value Q(s,a) for the action 'a' selected by the policy. This is the value we will train.
        selected_q = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1).squeeze(1)
        batch_size = actions.shape[0]
        # Determine the next state (s') by applying the selected action 'a'.
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        # Get all possible actions (a') and their costs c(s',a',s'') from the next state (s').
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        neighbor_cost = jnp.transpose(neighbor_cost, (1, 0))  # [batch_size, action_size]
        neighbor_valid_mask = jnp.isfinite(neighbor_cost)
        has_valid_neighbor = jnp.any(neighbor_valid_mask, axis=1)
        # Check if the next state (s') is a solved state.
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )
        selected_neighbors_solved = jnp.logical_or(
            selected_neighbors_solved, jnp.logical_not(has_valid_action)
        )

        # Preprocess the next states (s') for neural network input.
        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        # --- Target Q-Value Calculation (Bellman Optimality Equation) ---
        # Use the target Q-network (with frozen parameters `target_q_params`)
        # to get the Q-values for the next state, Q_target(s', a').
        # Using a separate target network stabilizes training.
        q = q_model.apply(
            target_q_params, preproc_neighbors, training=False
        )  # [minibatch_size, action_shape]
        q = jnp.nan_to_num(q, posinf=1e6, neginf=-1e6)
        # Invalidate actions that are not reachable from the next state.
        valid_neighbor_cost = jnp.where(neighbor_valid_mask, neighbor_cost, jnp.inf)

        # Modified for Q(s,a) = c(s,a) + min_a'(Q(s',a'))
        q_next = jnp.where(neighbor_valid_mask, q, jnp.inf)

        if use_double_dqn:
            q_online = q_model.apply(q_params, preproc_neighbors, training=False)
            q_online = jnp.nan_to_num(q_online, posinf=1e6, neginf=-1e6)
            q_online = jnp.where(jnp.isfinite(valid_neighbor_cost), q_online, jnp.inf)
            best_actions = jnp.argmin(q_online, axis=1)
            min_next_q = jnp.take_along_axis(q_next, best_actions[:, jnp.newaxis], axis=1).squeeze(
                1
            )
        else:
            min_next_q = jnp.min(q_next, axis=1)

        # Ensure non-negative future cost
        min_next_q = jnp.maximum(min_next_q, 0.0)
        min_next_q = jnp.where(has_valid_neighbor, min_next_q, 0.0)

        # Target entropy (confidence of the backup) over next-state distribution
        safe_temperature = jnp.maximum(temperature, 1e-8)
        scaled_next = -q_next / safe_temperature
        next_probs = jax.nn.softmax(scaled_next, axis=1)
        next_probs = jnp.where(neighbor_valid_mask, next_probs, 0.0)
        next_probs = next_probs / (jnp.sum(next_probs, axis=1, keepdims=True) + 1e-8)
        next_probs = jnp.where(has_valid_neighbor[:, jnp.newaxis], next_probs, 0.0)
        target_entropy = -jnp.sum(next_probs * jnp.log(jnp.clip(next_probs, a_min=1e-12)), axis=1)
        target_entropy = jnp.where(has_valid_neighbor, target_entropy, 0.0)
        # For solved states, entropy should be near zero (deterministic target)
        target_entropy = jnp.where(
            jnp.logical_or(solved, selected_neighbors_solved), 0.0, target_entropy
        )

        # Base case: If the next state (s') is the solution, the future cost is 0.
        # Q(s,a) = c(s,a) + (0 if solved else min Q(s',a'))
        selected_cost = jnp.take_along_axis(cost, actions[:, jnp.newaxis], axis=1).squeeze(1)
        selected_cost = jnp.where(has_valid_action, selected_cost, 0.0)
        target_q = selected_cost + jnp.where(selected_neighbors_solved, 0.0, min_next_q)
        # If the current state (s) was already solved, its Q-value should also be 0.
        target_q = jnp.where(solved, 0.0, target_q)
        target_q = jnp.where(has_valid_action, target_q, 0.0)

        # The 'diff' is the Temporal Difference (TD) error aligned with the training target
        diff = target_q - selected_q
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        diff = jnp.where(has_valid_action, diff, 0.0)
        # if the puzzle is already solved, the all q is 0
        return key, (
            solve_configs,
            states,
            target_q,
            actions,
            diff,
            entropy,
            max_entropy,
            target_entropy,
            max_entropy,
        )

    _, (
        solve_configs,
        states,
        target_q,
        actions,
        diff,
        entropy,
        max_entropy,
        target_entropy,
        target_max_entropy,
    ) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_q = target_q.reshape((-1, 1))
    actions = actions.reshape((-1, 1))
    diff = diff.reshape((-1,))
    entropy = entropy.reshape((-1,))
    max_entropy = max_entropy.reshape((-1,))
    target_entropy = target_entropy.reshape((-1,))
    target_max_entropy = target_max_entropy.reshape((-1,))
    cost = move_costs.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": actions,
        "diff": diff,
        "action_entropy": entropy,
        "action_entropy_max": max_entropy,
        "target_entropy": target_entropy,
        "target_entropy_max": target_max_entropy,
        "cost": cost,
    }


def _get_datasets_with_diffusion_distance(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStatesAndActions: Xtructurable,
    SolveConfigsAndStates: Xtructurable,
    q_model: QModelBase,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_double_dqn: bool = False,
):
    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]
    action_costs = shuffled_path["action_costs"].reshape((-1, 1))
    parent_indices = shuffled_path["parent_indices"]

    solve_configs_and_states_and_actions = SolveConfigsAndStatesAndActions(
        solveconfigs=solve_configs,
        states=states,
        actions=trajectory_actions,
    )
    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=move_costs,
        return_index=True,
        return_inverse=True,
    )
    target_q = move_costs[unique_uint32eds_idx][inverse_indices][
        :, jnp.newaxis
    ]  # [dataset_size, 1]

    # 1. Find state-wise minimal cost (Global Best Value Table)

    solve_configs_and_states = SolveConfigsAndStates(
        solveconfigs=solve_configs,
        states=states,
    )

    _, unique_state_idx, inverse_state_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=move_costs, # move_costs is used as initial cost estimate
        return_index=True,
        return_inverse=True,
    )
    # state_min_cost: best cost found for each state across all actions and trajectories
    state_min_cost = move_costs[unique_state_idx][inverse_state_indices][:, jnp.newaxis]

    # Propagate the improved Q values backwards along the trajectory
    dataset_size = target_q.shape[0]

    # Pad dataset with infinity to handle invalid parent pointers
    padded_q = jnp.pad(
        target_q, ((0, 1), (0, 0)), constant_values=jnp.inf
    )
    # Pad state_min_cost as well for lookup
    padded_state_min_cost = jnp.pad(
        state_min_cost, ((0, 1), (0, 0)), constant_values=jnp.inf
    )
    
    # Map -1 or out-of-bounds indices to the padded infinity value
    safe_parent_indices = jnp.where(
        (parent_indices < 0) | (parent_indices >= dataset_size), 
        dataset_size, 
        parent_indices
    )

    def body_fun(i, q):
        # q is padded [N+1, 1]
        current_q = q[:dataset_size]
        
        # Gather Q from parents (neighbors closer to goal)
        # Combine global optimal info (s' best) and trajectory info
        q_parents_optimal = padded_state_min_cost[safe_parent_indices] # [N, 1]
        q_parents_prop = q[safe_parent_indices] # [N, 1]
        
        q_parents = jnp.minimum(q_parents_optimal, q_parents_prop)
        
        # Bellman update: Q(s, a) <= c(s, a, s') + Q(s', a')
        # where (s', a') is the next state-action pair in the trajectory
        new_q = action_costs + q_parents
        
        improved_q = jnp.minimum(current_q, new_q)
        return q.at[:dataset_size].set(improved_q)

    # Iterate k_max times to propagate along the longest possible path
    final_padded_q = jax.lax.fori_loop(0, k_max, body_fun, padded_q)
    
    target_q = final_padded_q[:dataset_size]

    zeros = jnp.zeros_like(target_q)
    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": trajectory_actions,
        "diff": zeros,
        "action_entropy": zeros,
        "action_entropy_max": zeros,
        "target_entropy": zeros,
        "target_entropy_max": zeros,
        "cost": zeros,
    }


def _get_datasets_with_diffusion_distance_mixture(
    puzzle: Puzzle,
    preproc_fn: Callable,
    SolveConfigsAndStatesAndActions: Xtructurable,
    SolveConfigsAndStates: Xtructurable,
    q_model: QModelBase,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    k_max: int,
    shuffle_parallel: int,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_double_dqn: bool = False,
):
    trajectory_actions = shuffled_path["actions"].reshape((-1, 1))
    return_dict = _get_datasets_with_policy(
        puzzle,
        preproc_fn,
        q_model,
        minibatch_size,
        target_q_params,
        q_params,
        shuffled_path,
        key,
        temperature,
        td_error_clip,
        use_double_dqn,
    )
    solve_configs = return_dict["solveconfigs"]
    states = return_dict["states"]
    solve_configs_and_states_and_actions = SolveConfigsAndStatesAndActions(
        solveconfigs=solve_configs,
        states=states,
        actions=trajectory_actions,
    )
    cost = return_dict["cost"]
    target_q = return_dict["target_q"]
    
    # Prepare for propagation
    action_costs = shuffled_path["action_costs"].reshape((-1, 1))
    parent_indices = shuffled_path["parent_indices"]
    dataset_size = cost.shape[0]

    # Find unique states and broadcast the minimal cost to all duplicates
    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=cost,
        return_index=True,
        return_inverse=True,
    )
    cost = cost[unique_uint32eds_idx][inverse_indices][:, jnp.newaxis]  # [dataset_size, 1]

    # 1. Find state-wise minimal cost (Global Best Value Table)
    # This helps sharing optimal cost across different actions for the same state
    # Note: In Q-learning, Q(s,a) depends on 'a', but for distance-to-go metric, 
    # V(s) is a strong lower bound and good estimator.
    # We use this to enhance the back-propagation.
    
    # SolveConfigsAndStates structure needs to be defined/imported or created dynamically.
    # Since it is passed as an argument to other functions but not here, we might need to use the one from outer scope or create one.
    # However, we can reuse SolveConfigsAndStatesAndActions but ignore actions for grouping.
    # Or better, just create a temporary structure for state grouping.

    solve_configs_and_states = SolveConfigsAndStates(
        solveconfigs=solve_configs,
        states=states,
    )

    _, unique_state_idx, inverse_state_indices = xnp.unique_mask(
        val=solve_configs_and_states,
        key=cost,
        return_index=True,
        return_inverse=True,
    )
    # state_min_cost: best cost found for each state across all actions and trajectories
    state_min_cost = cost[unique_state_idx][inverse_state_indices][:, jnp.newaxis]

    # Propagate the improved cost values backwards along the trajectory
    # Pad dataset with infinity to handle invalid parent pointers
    padded_cost = jnp.pad(
        cost, ((0, 1), (0, 0)), constant_values=jnp.inf
    )
    # Pad state_min_cost as well for lookup
    padded_state_min_cost = jnp.pad(
        state_min_cost, ((0, 1), (0, 0)), constant_values=jnp.inf
    )
    
    # Map -1 or out-of-bounds indices to the padded infinity value
    safe_parent_indices = jnp.where(
        (parent_indices < 0) | (parent_indices >= dataset_size), 
        dataset_size, 
        parent_indices
    )

    def body_fun(i, c):
        # c is padded [N+1, 1]
        current_c = c[:dataset_size]
        
        # Gather optimal cost of the next state (s') from the global best table
        # This brings information from other trajectories
        c_parents_optimal = padded_state_min_cost[safe_parent_indices] # [N, 1]
        
        # Also consider the propagated cost within the current trajectory
        # This ensures consistency if trajectory update is faster/better
        c_parents_prop = c[safe_parent_indices]
        
        c_parents = jnp.minimum(c_parents_optimal, c_parents_prop)
        
        # Bellman update: C(s, a) <= step_cost + C(s', a')
        new_c = action_costs + c_parents
        
        improved_c = jnp.minimum(current_c, new_c)
        return c.at[:dataset_size].set(improved_c)

    # Iterate k_max times to propagate along the longest possible path
    final_padded_c = jax.lax.fori_loop(0, k_max, body_fun, padded_cost)
    
    cost = final_padded_c[:dataset_size]

    target_q = jnp.maximum(target_q, cost * 0.8 - 2.0)
    target_q = jnp.concatenate(
        (target_q, cost),
        axis=1,
    )  # [dataset_size, 2]
    return_dict["target_q"] = target_q
    actions = return_dict["actions"]
    actions = jnp.concatenate(
        (actions, trajectory_actions),
        axis=1,
    )  # [dataset_size, 2]
    return_dict["actions"] = actions
    return return_dict


def get_qlearning_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: QModelBase,
    dataset_size: int,
    k_max: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_double_dqn: bool = False,
    use_diffusion_distance: bool = False,
    use_diffusion_distance_mixture: Optional[float] = None,
    use_diffusion_distance_warmup: bool = False,
    diffusion_distance_warmup_steps: int = 0,
    non_backtracking_steps: int = 3,
):
    if non_backtracking_steps < 0:
        raise ValueError("non_backtracking_steps must be non-negative")
    non_backtracking_steps = int(non_backtracking_steps)

    if using_hindsight_target:
        assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
        # Calculate appropriate shuffle_parallel for hindsight sampling
        shuffle_parallel = int(min(math.ceil(dataset_size / k_max), dataset_minibatch_size))
        steps = math.ceil(dataset_size / (shuffle_parallel * k_max))
        if using_triangular_sampling:
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                False,
                non_backtracking_steps=non_backtracking_steps,
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                k_max,
                shuffle_parallel,
                False,
                non_backtracking_steps=non_backtracking_steps,
            )
    else:
        shuffle_parallel = int(min(math.ceil(dataset_size / k_max), dataset_minibatch_size))
        steps = math.ceil(dataset_size / (shuffle_parallel * k_max))
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            k_max,
            shuffle_parallel,
            False,
            non_backtracking_steps=non_backtracking_steps,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    base_get_datasets = partial(
        _get_datasets_with_policy,
        puzzle,
        preproc_fn,
        q_model,
        dataset_minibatch_size,
        temperature=temperature,
        td_error_clip=td_error_clip,
        use_double_dqn=use_double_dqn,
    )

    use_diffusion_features = use_diffusion_distance or use_diffusion_distance_mixture

    if use_diffusion_features:

        @xtructure_dataclass
        class SolveConfigsAndStatesAndActions:
            solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
            states: FieldDescriptor.scalar(dtype=puzzle.State)
            actions: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,))

        @xtructure_dataclass
        class SolveConfigsAndStates:
            solveconfigs: FieldDescriptor.scalar(dtype=puzzle.SolveConfig)
            states: FieldDescriptor.scalar(dtype=puzzle.State)

        if use_diffusion_distance_mixture:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance_mixture,
                puzzle,
                preproc_fn,
                SolveConfigsAndStatesAndActions,
                SolveConfigsAndStates,
                q_model,
                dataset_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
                td_error_clip=td_error_clip,
                use_double_dqn=use_double_dqn,
            )
        else:
            diffusion_get_datasets = partial(
                _get_datasets_with_diffusion_distance,
                puzzle,
                preproc_fn,
                SolveConfigsAndStatesAndActions,
                SolveConfigsAndStates,
                q_model,
                dataset_minibatch_size,
                k_max=k_max,
                shuffle_parallel=shuffle_parallel,
                temperature=temperature,
                td_error_clip=td_error_clip,
                use_double_dqn=use_double_dqn,
            )
    else:
        diffusion_get_datasets = base_get_datasets

    warmup_steps = max(int(diffusion_distance_warmup_steps), 0)
    warmup_enabled = use_diffusion_features and use_diffusion_distance_warmup and warmup_steps > 0

    def should_use_diffusion(step: int) -> bool:
        if not use_diffusion_features:
            return False
        if warmup_enabled:
            return step < warmup_steps
        return True

    def build_runner(dataset_extractor: Callable):
        @jax.jit
        def runner(
            target_q_params: Any,
            q_params: Any,
            key: chex.PRNGKey,
        ):
            def scan_fn(scan_key, _):
                scan_key, subkey = jax.random.split(scan_key)
                paths = jited_create_shuffled_path(subkey)
                return scan_key, paths

            key_inner, paths = jax.lax.scan(scan_fn, key, None, length=steps)
            for k, v in paths.items():
                paths[k] = v.flatten()[:dataset_size]
            flatten_dataset = dataset_extractor(
                target_q_params,
                q_params,
                paths,
                key_inner,
            )
            return flatten_dataset

        return runner

    default_runner = build_runner(base_get_datasets)
    diffusion_runner = build_runner(diffusion_get_datasets)

    if n_devices > 1:
        pmap_default_runner = jax.pmap(default_runner, in_axes=(None, None, 0))
        pmap_diffusion_runner = jax.pmap(diffusion_runner, in_axes=(None, None, 0))

        def get_datasets(target_q_params, q_params, key, step: int):
            keys = jax.random.split(key, n_devices)
            runner = pmap_diffusion_runner if should_use_diffusion(step) else pmap_default_runner
            return runner(target_q_params, q_params, keys)

        return get_datasets

    def single_device_get_datasets(target_q_params, q_params, key, step: int):
        runner = diffusion_runner if should_use_diffusion(step) else default_runner
        return runner(target_q_params, q_params, key)

    return single_device_get_datasets

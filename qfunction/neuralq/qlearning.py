import math
from functools import partial
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax
import xtructure.numpy as xnp
from puxle import Puzzle

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
    use_target_confidence_weighting: bool = False,
    use_target_sharpness_weighting: bool = False,
    target_sharpness_alpha: float = 1.0,
    using_priority_sampling: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    per_epsilon: float = 1e-6,
    loss_type: str = "mse",
    huber_delta: float = 0.1,
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
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        per_sample = loss_from_diff(diff, loss=loss_type, huber_delta=huber_delta)
        loss_value = jnp.mean(per_sample * weights)
        return loss_value, (new_params, diff)

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

        if using_priority_sampling:
            diff = dataset["diff"]
            # Sanitize TD errors to avoid NaN/Inf poisoning
            diff = jnp.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)

            # Calculate priorities based on TD error with strict positivity
            priorities = jnp.abs(diff) + per_epsilon
            priorities = jnp.clip(priorities, a_min=1e-12)

            # Stable sampling probabilities in log-space: p_i ∝ priorities^alpha
            logp = per_alpha * jnp.log(priorities)
            logp = logp - jnp.max(logp)
            sampling_probs = jnp.exp(logp)
            sampling_probs = sampling_probs / (jnp.sum(sampling_probs) + 1e-12)

            # Stable importance sampling weights in log-space; max-normalized to 1
            clipped_probs = jnp.clip(sampling_probs, a_min=1e-12)
            log_w = -per_beta * (jnp.log(data_size) + jnp.log(clipped_probs))
            log_w = log_w - jnp.max(log_w)
            is_weights = jnp.exp(log_w)
            loss_weights = is_weights
        else:
            loss_weights = jnp.ones(data_size)

        if use_target_confidence_weighting:
            cost = dataset["cost"]
            cost_weights = 1.0 / jnp.sqrt(jnp.maximum(cost, 1.0))
            cost_weights = cost_weights / jnp.mean(cost_weights)
            loss_weights = loss_weights * cost_weights

        if use_target_sharpness_weighting:
            # Prefer target entropy if available; fallback to behavior entropy
            entropy = None
            max_entropy = None
            if "target_entropy" in dataset:
                entropy = dataset["target_entropy"]
                max_entropy = dataset.get("target_entropy_max", None)
            elif "action_entropy" in dataset:
                entropy = dataset.get("action_entropy", None)
                max_entropy = dataset.get("action_entropy_max", None)

            if entropy is not None:
                if max_entropy is None:
                    # Fallback: approximate with log(action_size) if available
                    action_size = jnp.max(actions) + 1 if "actions" in dataset else 1
                    max_entropy = jnp.log(jnp.maximum(action_size.astype(jnp.float32), 1.0))
                normalized_entropy = entropy / (max_entropy + 1e-8)
            else:
                normalized_entropy = jnp.zeros(data_size)

            sharpness = 1.0 - normalized_entropy
            sharp_weights = 1.0 + target_sharpness_alpha * sharpness
            sharp_weights = sharp_weights / (jnp.mean(sharp_weights) + 1e-8)
            loss_weights = loss_weights * sharp_weights

        if not using_priority_sampling:
            loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            solveconfigs, states, target_q, actions, weights = batched_dataset
            (loss, (q_params, diff)), grads = jax.value_and_grad(qlearning_loss, has_aux=True)(
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
            return (q_params, opt_state), (loss, grad_magnitude_mean, diff)

        # Repeat training loop for replay_ratio times with reshuffling
        def replay_loop(carry, replay_key):
            q_params, opt_state = carry

            # Reshuffle the batch indices for each replay iteration
            if using_priority_sampling:
                # For priority sampling, resample based on priorities
                batch_indexs_replay = jax.random.choice(
                    replay_key,
                    jnp.arange(data_size),
                    shape=(batch_size * minibatch_size,),
                    p=sampling_probs,
                    replace=True,
                )
                loss_weights_replay = is_weights
            else:
                # For uniform sampling, create new permutation
                key_perm_replay, key_fill_replay = jax.random.split(replay_key)
                batch_indexs_replay = jnp.concatenate(
                    [
                        jax.random.permutation(key_perm_replay, jnp.arange(data_size)),
                        jax.random.randint(
                            key_fill_replay,
                            (batch_size * minibatch_size - data_size,),
                            0,
                            data_size,
                        ),
                    ],
                    axis=0,
                )
                loss_weights_replay = loss_weights

            batch_indexs_replay = jnp.reshape(batch_indexs_replay, (batch_size, minibatch_size))

            # Create new batches with reshuffled indices
            batched_solveconfigs_replay = xnp.take(solveconfigs, batch_indexs_replay, axis=0)
            batched_states_replay = xnp.take(states, batch_indexs_replay, axis=0)
            batched_target_q_replay = jnp.take(target_q, batch_indexs_replay, axis=0)
            batched_actions_replay = jnp.take(actions, batch_indexs_replay, axis=0)
            batched_weights_replay = jnp.take(loss_weights_replay, batch_indexs_replay, axis=0)
            # Normalize weights per batch to prevent scale drift
            batched_weights_replay = batched_weights_replay / (
                jnp.mean(batched_weights_replay, axis=1, keepdims=True) + 1e-8
            )

            (q_params, opt_state), (losses, grad_magnitude_means, diffs) = jax.lax.scan(
                train_loop,
                (q_params, opt_state),
                (
                    batched_solveconfigs_replay,
                    batched_states_replay,
                    batched_target_q_replay,
                    batched_actions_replay,
                    batched_weights_replay,
                ),
            )
            return (q_params, opt_state), (losses, grad_magnitude_means, diffs)

        # Generate separate keys for each replay iteration
        replay_keys = jax.random.split(key, replay_ratio)
        (q_params, opt_state), (losses, grad_magnitude_means, diffs) = jax.lax.scan(
            replay_loop,
            (q_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        diffs = diffs.reshape(-1)
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
        )

    if n_devices > 1:

        def pmap_qlearning(key, dataset, q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude, diffs) = jax.pmap(
                qlearning, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, q_params, opt_state)
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            diffs = diffs.reshape(-1)
            return qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude, diffs

        return pmap_qlearning
    else:
        return jax.jit(qlearning)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
) -> chex.Array:
    # Sanitize inputs
    q_values = jnp.nan_to_num(q_values, posinf=1e6, neginf=-1e6)
    mask = jnp.isfinite(q_values)

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


def log_pi(probs: chex.Array) -> chex.Array:
    return jnp.log(jnp.clip(probs, a_min=1e-12))


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
    use_munchausen: bool = False,
    td_error_clip: Optional[float] = None,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states, move_costs = vals
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
        q_sum_cost = q_values + cost

        # Select an action 'a' probabilistically using a Boltzmann (softmax) exploration policy.
        # Actions with lower Q-values (lower cost-to-go) are more likely to be chosen.
        # Epsilon-greedy exploration is also mixed in.
        probs = boltzmann_action_selection(q_sum_cost, temperature=temperature)
        # Action entropy per state (measure of policy sharpness)
        log_pi_probs = log_pi(probs)
        entropy = -jnp.sum(probs * log_pi_probs, axis=1)
        if use_munchausen:
            cost = cost - 0.1 * jnp.clip(temperature * log_pi_probs, a_min=-3.0)
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
        # Check if the next state (s') is a solved state.
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
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
        q = jnp.where(jnp.isfinite(neighbor_cost), q, jnp.inf)
        q = jnp.nan_to_num(q, posinf=1e6, neginf=-1e6)
        # Calculate the target Q-value using the Bellman Optimality Equation:
        # target_Q(s, a) = c(s, a, s') + min_{a'} Q_target(s', a')
        # This represents the optimal cost-to-go from s'.
        q_sum_cost = q + cost  # [batch_size, action_size]
        # Clamp to ensure non-negative targets (costs should be non-negative)
        q_sum_cost = jnp.maximum(q_sum_cost, cost)
        # Target entropy (confidence of the backup) over next-state distribution
        safe_temperature = jnp.maximum(temperature, 1e-8)
        scaled_next = -q_sum_cost / safe_temperature
        next_probs = jax.nn.softmax(scaled_next, axis=1)
        log_pi_next_probs = log_pi(next_probs)
        target_entropy = -jnp.sum(next_probs * log_pi_next_probs, axis=1)
        # For solved states, entropy should be near zero (deterministic target)
        target_entropy = jnp.where(
            jnp.logical_or(solved, selected_neighbors_solved), 0.0, target_entropy
        )
        if use_munchausen:
            target_q = jnp.sum(next_probs * (q_sum_cost + temperature * log_pi_next_probs), axis=1)
        else:
            target_q = jnp.min(q_sum_cost, axis=1)

        # Base case: If the next state (s') is the solution, the future cost is 0.
        target_q = jnp.where(selected_neighbors_solved, 0.0, target_q)
        # If the current state (s) was already solved, its Q-value should also be 0.
        target_q = jnp.where(solved, 0.0, target_q)

        # The 'diff' is the Temporal Difference (TD) error aligned with the training target
        diff = target_q - selected_q
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
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
            move_costs,
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
        cost,
    ) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_q = target_q.reshape((-1,))
    actions = actions.reshape((-1,))
    diff = diff.reshape((-1,))
    entropy = entropy.reshape((-1,))
    max_entropy = max_entropy.reshape((-1,))
    target_entropy = target_entropy.reshape((-1,))
    target_max_entropy = target_max_entropy.reshape((-1,))
    cost = cost.reshape((-1,))

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


def _get_datasets_with_trajectory(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: QModelBase,
    minibatch_size: int,
    target_q_params: Any,
    q_params: Any,
    shuffled_path: dict[str, chex.Array],
    key: chex.PRNGKey,
    td_error_clip: Optional[float] = None,
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    actions = shuffled_path["actions"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = solve_configs.reshape((-1, minibatch_size))
    minibatched_states = states.reshape((-1, minibatch_size))
    minibatched_actions = actions.reshape((-1, minibatch_size))
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states, actions, move_costs = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]

        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        neighbor_cost = jnp.transpose(neighbor_cost, (1, 0))  # [batch_size, action_size]
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )

        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        q = q_model.apply(
            target_q_params, preproc_neighbors, training=False
        )  # [minibatch_size, action_shape]
        q = jnp.where(jnp.isfinite(neighbor_cost), q, jnp.inf)
        q_sum_cost = q + cost
        # Clamp to ensure non-negative targets (costs should be non-negative)
        q_sum_cost = jnp.maximum(q_sum_cost, cost)
        min_q_sum_cost = jnp.min(q_sum_cost, axis=1)

        target_q = jnp.where(selected_neighbors_solved, 0.0, min_q_sum_cost)
        target_q = jnp.where(solved, 0.0, target_q)

        diff = jnp.zeros_like(target_q)
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        # if the puzzle is already solved, the all q is 0
        return key, (solve_configs, states, target_q, actions, diff, move_costs)

    _, (solve_configs, states, target_q, actions, diff, cost) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (
            minibatched_solve_configs,
            minibatched_states,
            minibatched_actions,
            minibatched_move_costs,
        ),
    )

    solve_configs = solve_configs.reshape((-1,))
    states = states.reshape((-1,))
    target_q = target_q.reshape((-1,))
    actions = actions.reshape((-1,))
    diff = diff.reshape((-1,))
    cost = cost.reshape((-1,))

    return {
        "solveconfigs": solve_configs,
        "states": states,
        "target_q": target_q,
        "actions": actions,
        "diff": diff,
        "cost": cost,
    }


def get_qlearning_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    q_model: QModelBase,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    with_policy: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    use_munchausen: bool = False,
    td_error_clip: Optional[float] = None,
):
    if using_hindsight_target:
        assert not puzzle.fixed_target, "Fixed target is not supported for hindsight target"
        # Calculate appropriate shuffle_parallel for hindsight sampling
        shuffle_parallel = int(
            min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
        )
        steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
        if using_triangular_sampling:
            create_shuffled_path_fn = partial(
                create_hindsight_target_triangular_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
                False,
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
                False,
            )
    else:
        with_policy = True  # if not using hindsight target, must use policy for training
        shuffle_parallel = int(
            min(math.ceil(dataset_size / shuffle_length), dataset_minibatch_size)
        )
        steps = math.ceil(dataset_size / (shuffle_parallel * shuffle_length))
        create_shuffled_path_fn = partial(
            create_target_shuffled_path,
            puzzle,
            shuffle_length,
            shuffle_parallel,
            False,
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    if with_policy:
        jited_get_datasets = jax.jit(
            partial(
                _get_datasets_with_policy,
                puzzle,
                preproc_fn,
                q_model,
                dataset_minibatch_size,
                temperature=temperature,
                use_munchausen=use_munchausen,
                td_error_clip=td_error_clip,
            )
        )
    else:
        jited_get_datasets = jax.jit(
            partial(
                _get_datasets_with_trajectory,
                puzzle,
                preproc_fn,
                q_model,
                dataset_minibatch_size,
                use_munchausen=use_munchausen,
                td_error_clip=td_error_clip,
            )
        )

    @jax.jit
    def get_datasets(
        target_q_params: Any,
        q_params: Any,
        key: chex.PRNGKey,
    ):
        def scan_fn(key, _):
            key, subkey = jax.random.split(key)
            paths = jited_create_shuffled_path(subkey)
            return key, paths

        key, paths = jax.lax.scan(scan_fn, key, None, length=steps)
        for k, v in paths.items():
            paths[k] = v.flatten()[:dataset_size]
        flatten_dataset = jited_get_datasets(target_q_params, q_params, paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(target_q_params, q_params, key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(None, None, 0))(
                target_q_params, q_params, keys
            )
            return datasets

        return pmap_get_datasets
    else:
        return get_datasets

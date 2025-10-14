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
        q_values_at_actions = jnp.take_along_axis(
            q_values, actions, axis=1
        )  # [batch_size, minibatch_size, 1 or 2]
        diff = target_qs - q_values_at_actions  # [batch_size, minibatch_size, 1 or 2]
        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)
        per_sample = loss_from_diff(diff, loss=loss_type, huber_delta=huber_delta)
        loss_value = jnp.mean(per_sample * weights[:, jnp.newaxis])
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

        loss_weights = jnp.ones(data_size)
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

            (q_params, opt_state), (losses, grad_magnitude_means, diffs) = jax.lax.scan(
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
            return (q_params, opt_state), (losses, grad_magnitude_means, diffs)

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
    move_costs_tm1 = shuffled_path["move_costs_tm1"]

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
        q_sum_cost = q_values + cost

        # Select an action 'a' probabilistically using a Boltzmann (softmax) exploration policy.
        # Actions with lower Q-values (lower cost-to-go) are more likely to be chosen.
        # Epsilon-greedy exploration is also mixed in.
        probs = boltzmann_action_selection(q_sum_cost, temperature=temperature)
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
        q = jnp.nan_to_num(q, posinf=1e6, neginf=-1e6)
        # Invalidate actions that are not reachable from the next state.
        valid_neighbor_cost = jnp.where(jnp.isfinite(neighbor_cost), neighbor_cost, jnp.inf)
        q = jnp.where(jnp.isfinite(valid_neighbor_cost), q, jnp.inf)
        # Calculate the target Q-value using the constrained Bellman equation:
        # target_Q(s, a) = min_{a'} [ c(s', a') + Q_target(s', a') ].
        # The immediate cost of (s, a) is excluded by construction (J(s, a) = J(N(s, a))).
        q_sum_cost = valid_neighbor_cost + q  # [batch_size, action_size]
        # Clamp to ensure non-negative targets while respecting the next-state costs.
        q_sum_cost = jnp.maximum(q_sum_cost, valid_neighbor_cost)
        if use_double_dqn:
            q_online = q_model.apply(q_params, preproc_neighbors, training=False)
            q_online = jnp.nan_to_num(q_online, posinf=1e6, neginf=-1e6)
            q_online = jnp.where(jnp.isfinite(valid_neighbor_cost), q_online, jnp.inf)
            online_q_sum_cost = valid_neighbor_cost + q_online
            online_q_sum_cost = jnp.maximum(online_q_sum_cost, valid_neighbor_cost)
            best_actions = jnp.argmin(online_q_sum_cost, axis=1)
            min_q_sum_cost = jnp.take_along_axis(q_sum_cost, best_actions[:, jnp.newaxis], axis=1)[
                :, 0
            ]
        else:
            min_q_sum_cost = jnp.min(q_sum_cost, axis=1)
        # Target entropy (confidence of the backup) over next-state distribution
        safe_temperature = jnp.maximum(temperature, 1e-8)
        scaled_next = -q_sum_cost / safe_temperature
        next_probs = jax.nn.softmax(scaled_next, axis=1)
        next_probs = next_probs / (jnp.sum(next_probs, axis=1, keepdims=True) + 1e-8)
        target_entropy = -jnp.sum(next_probs * jnp.log(jnp.clip(next_probs, a_min=1e-12)), axis=1)
        # For solved states, entropy should be near zero (deterministic target)
        target_entropy = jnp.where(
            jnp.logical_or(solved, selected_neighbors_solved), 0.0, target_entropy
        )

        # Base case: If the next state (s') is the solution, the future cost is 0.
        target_q = jnp.where(selected_neighbors_solved, 0.0, min_q_sum_cost)
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
    cost = move_costs_tm1.reshape((-1,))

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
    _, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
        val=solve_configs_and_states_and_actions,
        key=cost,
        return_index=True,
        return_inverse=True,
    )
    cost = cost[unique_uint32eds_idx][inverse_indices][:, jnp.newaxis]  # [dataset_size, 1]
    target_q = jnp.maximum(target_q, cost)
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
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
    temperature: float = 1.0 / 3.0,
    td_error_clip: Optional[float] = None,
    use_double_dqn: bool = False,
    use_diffusion_distance: bool = False,
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

    if use_diffusion_distance:

        @xtructure_dataclass
        class SolveConfigsAndStatesAndActions:
            solveconfigs: FieldDescriptor[puzzle.SolveConfig]
            states: FieldDescriptor[puzzle.State]
            actions: FieldDescriptor[jnp.uint8, (1,)]

        jited_get_datasets = jax.jit(
            partial(
                _get_datasets_with_diffusion_distance,
                puzzle,
                preproc_fn,
                SolveConfigsAndStatesAndActions,
                q_model,
                dataset_minibatch_size,
                temperature=temperature,
                td_error_clip=td_error_clip,
                use_double_dqn=use_double_dqn,
            )
        )
    else:
        jited_get_datasets = jax.jit(
            partial(
                _get_datasets_with_policy,
                puzzle,
                preproc_fn,
                q_model,
                dataset_minibatch_size,
                temperature=temperature,
                td_error_clip=td_error_clip,
                use_double_dqn=use_double_dqn,
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

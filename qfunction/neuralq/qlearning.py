import math
from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax
from puxle import Puzzle

from helpers.sampling import (
    create_hindsight_target_shuffled_path,
    create_hindsight_target_triangular_shuffled_path,
    create_target_shuffled_path,
)
from qfunction.neuralq.neuralq_base import QModelBase


def qlearning_builder(
    minibatch_size: int,
    q_fn: QModelBase,
    optimizer: optax.GradientTransformation,
    importance_sampling: int = True,
    importance_sampling_alpha: float = 0.5,
    importance_sampling_beta: float = 0.1,
    importance_sampling_eps: float = 1.0,
    n_devices: int = 1,
    use_target_confidence_weighting: bool = False,
):
    def qlearning_loss(
        q_params: Any,
        preproc: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        weights: chex.Array,
    ):
        q_values, variable_updates = q_fn.apply(
            q_params, preproc, training=True, mutable=["batch_stats"]
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name="devices")
        new_params = {"params": q_params["params"], "batch_stats": variable_updates["batch_stats"]}
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        diff = target_qs.squeeze() - q_values_at_actions.squeeze()
        loss = jnp.mean(jnp.square(diff) * weights)
        return loss, new_params

    def qlearning(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        q_params: Any,
        opt_state: optax.OptState,
    ):
        """
        Q-learning is a heuristic for the sliding puzzle problem.
        """
        preproc = dataset["preproc"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        diff = dataset["diff"]
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        if importance_sampling:
            # Calculate sampling probabilities based on diff (error) for importance sampling
            # Higher diff values get higher probability (similar to PER - Prioritized Experience Replay)
            abs_diff = jnp.abs(diff)
            sampling_weights = jnp.power(
                abs_diff + importance_sampling_eps, importance_sampling_alpha
            )
            sampling_probs = sampling_weights / jnp.sum(sampling_weights)
            loss_weights = jnp.power(data_size * sampling_probs, -importance_sampling_beta)
            loss_weights = loss_weights / jnp.max(loss_weights)

            # Sample indices based on the calculated probabilities
            batch_indexs = jax.random.choice(
                key,
                jnp.arange(data_size),
                shape=(batch_size * minibatch_size,),
                replace=False,
                p=sampling_probs,
            )
        else:
            batch_indexs = jnp.concatenate(
                [
                    jax.random.permutation(key, jnp.arange(data_size)),
                    jax.random.randint(
                        key, (batch_size * minibatch_size - data_size,), 0, data_size
                    ),
                ],
                axis=0,
            )  # [batch_size * minibatch_size]
            loss_weights = jnp.ones_like(batch_indexs)
        if use_target_confidence_weighting:
            cost = dataset["cost"]
            cost_weights = 1.0 / jnp.maximum(cost, 1.0)
            cost_weights = cost_weights / jnp.mean(cost_weights)
            loss_weights = loss_weights * cost_weights
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_preproc = jnp.take(preproc, batch_indexs, axis=0)
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)
        batched_weights = jnp.take(loss_weights, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            preproc, target_q, actions, weights = batched_dataset
            (loss, q_params), grads = jax.value_and_grad(qlearning_loss, has_aux=True)(
                q_params,
                preproc,
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
            return (q_params, opt_state), (loss, grad_magnitude_mean)

        (q_params, opt_state), (losses, grad_magnitude_means) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (batched_preproc, batched_target_q, batched_actions, batched_weights),
        )
        loss = jnp.mean(losses)
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
        )

    if n_devices > 1:

        def pmap_qlearning(key, dataset, q_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude,) = jax.pmap(
                qlearning, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, q_params, opt_state)
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            return qfunc_params, opt_state, loss, grad_magnitude, weight_magnitude

        return pmap_qlearning
    else:
        return jax.jit(qlearning)


def boltzmann_action_selection(
    q_values: chex.Array,
    temperature: float = 1.0 / 3.0,
    epsilon: float = 0.1,
    mask: chex.Array = None,
) -> chex.Array:
    q_values = -q_values / temperature
    probs = jnp.exp(q_values)
    if mask is not None:
        probs = jnp.where(mask, probs, 0.0)
    else:
        mask = jnp.ones_like(probs)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)
    uniform_prob = mask.astype(jnp.float32) / jnp.sum(mask, axis=1, keepdims=True)
    probs = probs * (1 - epsilon) + uniform_prob * epsilon
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
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_states = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), states
    )
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size, *move_costs.shape[1:]))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states, move_costs = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        q_values, _ = q_model.apply(q_params, preproc, training=False, mutable=["batch_stats"])
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]
        mask = jnp.isfinite(jnp.transpose(cost, (1, 0)))

        probs = boltzmann_action_selection(q_values, temperature=temperature, mask=mask)
        idxs = jnp.arange(q_values.shape[1])  # action_size
        actions = jax.vmap(lambda key, p: jax.random.choice(key, idxs, p=p), in_axes=(0, 0))(
            jax.random.split(subkey, q_values.shape[0]), probs
        )
        selected_q = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1).squeeze(1)

        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        selected_costs = jnp.take_along_axis(cost, actions[jnp.newaxis, :], axis=0).squeeze(0)
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )

        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        q, _ = q_model.apply(
            target_q_params, preproc_neighbors, training=False, mutable=["batch_stats"]
        )  # [minibatch_size, action_shape]
        mask = jnp.isfinite(jnp.transpose(neighbor_cost, (1, 0)))
        q = jnp.where(mask, q, jnp.inf)
        min_q = jnp.min(q, axis=1)
        target_q = jnp.maximum(min_q, 0.0) + selected_costs
        target_q = jnp.where(selected_neighbors_solved, selected_costs, target_q)
        target_q = jnp.where(solved, 0.0, target_q)

        diff = target_q - selected_q
        # if the puzzle is already solved, the all q is 0
        return key, (preproc, target_q, actions, diff, move_costs)

    _, (preproc, target_q, actions, diff, cost) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (minibatched_solve_configs, minibatched_states, minibatched_move_costs),
    )

    preproc = preproc.reshape((-1, *preproc.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    diff = diff.reshape((-1, *diff.shape[2:]))
    cost = cost.reshape((-1, *cost.shape[2:]))

    return {
        "preproc": preproc,
        "target_q": target_q,
        "actions": actions,
        "diff": diff,
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
):
    solve_configs = shuffled_path["solve_configs"]
    states = shuffled_path["states"]
    actions = shuffled_path["actions"]
    move_costs = shuffled_path["move_costs"]

    minibatched_solve_configs = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), solve_configs
    )
    minibatched_states = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), states
    )
    minibatched_actions = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), actions
    )
    minibatched_move_costs = move_costs.reshape((-1, minibatch_size, *move_costs.shape[1:]))

    def get_minibatched_datasets(key, vals):
        key, subkey = jax.random.split(key)
        solve_configs, states, actions, move_costs = vals
        solved = puzzle.batched_is_solved(solve_configs, states, multi_solve_config=True)

        preproc = jax.vmap(preproc_fn)(solve_configs, states)
        neighbors, cost = puzzle.batched_get_neighbours(
            solve_configs, states, filleds=jnp.ones(minibatch_size), multi_solve_config=True
        )  # [action_size, batch_size] [action_size, batch_size]

        batch_size = actions.shape[0]
        selected_neighbors = jax.tree_util.tree_map(
            lambda x: x[actions, jnp.arange(batch_size), :],
            neighbors,
        )
        selected_costs = jnp.take_along_axis(cost, actions[jnp.newaxis, :], axis=0).squeeze(0)
        _, neighbor_cost = puzzle.batched_get_neighbours(
            solve_configs,
            selected_neighbors,
            filleds=jnp.ones(minibatch_size),
            multi_solve_config=True,
        )  # [action_size, batch_size] [action_size, batch_size]
        selected_neighbors_solved = puzzle.batched_is_solved(
            solve_configs, selected_neighbors, multi_solve_config=True
        )

        preproc_neighbors = jax.vmap(preproc_fn, in_axes=(0, 0))(solve_configs, selected_neighbors)

        q, _ = q_model.apply(
            target_q_params, preproc_neighbors, training=False, mutable=["batch_stats"]
        )  # [minibatch_size, action_shape]
        mask = jnp.isfinite(jnp.transpose(neighbor_cost, (1, 0)))
        q = jnp.where(mask, q, jnp.inf)
        min_q = jnp.min(q, axis=1)
        target_q = jnp.maximum(min_q, 0.0) + selected_costs
        target_q = jnp.where(selected_neighbors_solved, selected_costs, target_q)
        target_q = jnp.where(solved, 0.0, target_q)

        diff = jnp.zeros_like(target_q)
        # if the puzzle is already solved, the all q is 0
        return key, (preproc, target_q, actions, diff, move_costs)

    _, (preproc, target_q, actions, diff, cost) = jax.lax.scan(
        get_minibatched_datasets,
        key,
        (
            minibatched_solve_configs,
            minibatched_states,
            minibatched_actions,
            minibatched_move_costs,
        ),
    )

    preproc = preproc.reshape((-1, *preproc.shape[2:]))
    target_q = target_q.reshape((-1, *target_q.shape[2:]))
    actions = actions.reshape((-1, *actions.shape[2:]))
    diff = diff.reshape((-1, *diff.shape[2:]))
    cost = cost.reshape((-1, *cost.shape[2:]))

    return {
        "preproc": preproc,
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
            )
        else:
            create_shuffled_path_fn = partial(
                create_hindsight_target_shuffled_path,
                puzzle,
                shuffle_length,
                shuffle_parallel,
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
        )

    jited_create_shuffled_path = jax.jit(create_shuffled_path_fn)

    jited_get_datasets = jax.jit(
        partial(
            _get_datasets_with_policy if with_policy else _get_datasets_with_trajectory,
            puzzle,
            preproc_fn,
            q_model,
            dataset_minibatch_size,
            temperature=temperature if with_policy else None,
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
        paths = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:]))[:dataset_size], paths
        )

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

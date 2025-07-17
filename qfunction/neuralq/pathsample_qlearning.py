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
from neural_util.losses import quantile_weighted_huber_loss
from qfunction.neuralq.neuralq_base import QModelBase


def pathsample_qlearning_builder(
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
    ):
        q_values, variable_updates = q_fn.apply(
            q_params, preproc, training=True, mutable=["batch_stats"]
        )
        if n_devices > 1:
            variable_updates = jax.lax.pmean(variable_updates, axis_name="devices")
        new_params = {"params": q_params["params"], "batch_stats": variable_updates["batch_stats"]}
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        loss = quantile_weighted_huber_loss(
            q_values_at_actions.squeeze(), target_qs.squeeze(), quantile=0.2
        )
        return loss, (new_params, q_values_at_actions)

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
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        batch_indexs = jnp.concatenate(
            [
                jax.random.permutation(key, jnp.arange(data_size)),
                jax.random.randint(key, (batch_size * minibatch_size - data_size,), 0, data_size),
            ],
            axis=0,
        )  # [batch_size * minibatch_size]
        batch_indexs = jnp.reshape(batch_indexs, (batch_size, minibatch_size))

        batched_preproc = jnp.take(preproc, batch_indexs, axis=0)
        batched_target_q = jnp.take(target_q, batch_indexs, axis=0)
        batched_actions = jnp.take(actions, batch_indexs, axis=0)

        def train_loop(carry, batched_dataset):
            q_params, opt_state = carry
            preproc, target_q, actions = batched_dataset
            (loss, (q_params, q_values_at_actions)), grads = jax.value_and_grad(
                qlearning_loss, has_aux=True
            )(
                q_params,
                preproc,
                actions,
                target_q,
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
            return (q_params, opt_state), (loss, grad_magnitude_mean, q_values_at_actions)

        (q_params, opt_state), (losses, grad_magnitude_means, q_values_at_actions) = jax.lax.scan(
            train_loop,
            (q_params, opt_state),
            (batched_preproc, batched_target_q, batched_actions),
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
            q_values_at_actions,
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
                q_values_at_actions,
            ) = jax.pmap(qlearning, in_axes=(0, 0, None, None), axis_name="devices")(
                keys, dataset, q_params, opt_state
            )
            qfunc_params = jax.tree_util.tree_map(lambda xs: xs[0], qfunc_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            grad_magnitude = jnp.mean(grad_magnitude)
            weight_magnitude = jnp.mean(weight_magnitude)
            return (
                qfunc_params,
                opt_state,
                loss,
                grad_magnitude,
                weight_magnitude,
                q_values_at_actions,
            )

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


def _get_datasets_with_pathsample(
    puzzle: Puzzle,
    preproc_fn: Callable,
    minibatch_size: int,
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
        target_q = jnp.maximum(move_costs - 1.0, 0.0)
        target_q = jnp.where(solved, 0.0, target_q)

        return key, (preproc, target_q, actions)

    _, (preproc, target_q, actions) = jax.lax.scan(
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

    return {
        "preproc": preproc,
        "target_q": target_q,
        "actions": actions,
    }


def get_qlearning_pathsample_dataset_builder(
    puzzle: Puzzle,
    preproc_fn: Callable,
    dataset_size: int,
    shuffle_length: int,
    dataset_minibatch_size: int,
    using_hindsight_target: bool = True,
    using_triangular_sampling: bool = False,
    n_devices: int = 1,
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
            _get_datasets_with_pathsample,
            puzzle,
            preproc_fn,
            dataset_minibatch_size,
        )
    )

    @jax.jit
    def get_datasets(
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

        flatten_dataset = jited_get_datasets(paths, key)
        return flatten_dataset

    if n_devices > 1:

        def pmap_get_datasets(key):
            keys = jax.random.split(key, n_devices)
            datasets = jax.pmap(get_datasets, in_axes=(0))(keys)
            return datasets

        return pmap_get_datasets
    else:
        return get_datasets

import math
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.sampling import minibatch_datasets
from train_util.util import (
    apply_with_conditional_batch_stats,
    build_new_params_from_updates,
)


def heuristic_train_builder(
    minibatch_size: int,
    heuristic_model: DistanceModel | DistanceHLGModel,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: Optional[dict[str, Any]] = None,
    replay_ratio: int = 1,
):
    def heuristic_train_loss(
        heuristic_params: Any,
        solveconfigs: chex.Array,
        states: chex.Array,
        target_heuristic: chex.Array,
        weights: chex.Array,
        key: chex.PRNGKey,
    ):
        # Preprocess during training
        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        (per_sample_loss, aux), variable_updates = apply_with_conditional_batch_stats(
            heuristic_model.apply,
            heuristic_params,
            preproc,
            target_heuristic,
            training=True,
            n_devices=n_devices,
            method=heuristic_model.train_loss,
            loss_type=loss_type,
            loss_args=loss_args,
            rngs={"params": key},
        )
        new_params = build_new_params_from_updates(heuristic_params, variable_updates)
        loss_value = jnp.mean(per_sample_loss.squeeze() * weights)
        return loss_value, (new_params, aux)

    def heuristic_train(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        heuristic_params: Any,
        opt_state: optax.OptState,
    ):
        """
        DAVI is a heuristic for the sliding puzzle problem.
        """
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_heuristic = dataset["target_heuristic"]
        data_size = target_heuristic.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            heuristic_params, opt_state, key = carry
            step_key, key = jax.random.split(key)
            solveconfigs, states, target_heuristic, weights = batched_dataset
            (loss, (heuristic_params, aux)), grads = jax.value_and_grad(
                heuristic_train_loss, has_aux=True
            )(
                heuristic_params,
                solveconfigs,
                states,
                target_heuristic,
                weights,
                step_key,
            )
            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")
            updates, opt_state = optimizer.update(grads, opt_state, params=heuristic_params)
            heuristic_params = optax.apply_updates(heuristic_params, updates)
            return (heuristic_params, opt_state, key), (loss, aux)

        # Repeat training loop for replay_ratio iterations with reshuffling
        def replay_loop(carry, replay_key):
            heuristic_params, opt_state = carry

            key_replay, key_train = jax.random.split(replay_key)
            (
                batched_solveconfigs_replay,
                batched_states_replay,
                batched_target_heuristic_replay,
                batched_weights_replay,
            ) = minibatch_datasets(
                solveconfigs,
                states,
                target_heuristic,
                loss_weights,
                data_size=data_size,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                key=key_replay,
            )

            (heuristic_params, opt_state, _), (losses, auxs) = jax.lax.scan(
                train_loop,
                (heuristic_params, opt_state, key_train),
                (
                    batched_solveconfigs_replay,
                    batched_states_replay,
                    batched_target_heuristic_replay,
                    batched_weights_replay,
                ),
            )
            return (heuristic_params, opt_state), (losses, auxs)

        # Generate keys for replay iterations
        replay_keys = jax.random.split(key, replay_ratio)
        (heuristic_params, opt_state), (losses, auxs) = jax.lax.scan(
            replay_loop,
            (heuristic_params, opt_state),
            replay_keys,
        )
        loss = jnp.mean(losses)
        return (
            heuristic_params,
            opt_state,
            loss,
            auxs,
        )

    if n_devices > 1:

        def pmap_heuristic_train(key, dataset, heuristic_params, opt_state):
            keys = jax.random.split(key, n_devices)
            (heuristic_params, opt_state, loss, auxs,) = jax.pmap(
                heuristic_train, in_axes=(0, 0, None, None), axis_name="devices"
            )(keys, dataset, heuristic_params, opt_state)
            heuristic_params = jax.tree_util.tree_map(lambda xs: xs[0], heuristic_params)
            opt_state = jax.tree_util.tree_map(lambda xs: xs[0], opt_state)
            loss = jnp.mean(loss)
            return (
                heuristic_params,
                opt_state,
                loss,
                auxs,
            )

        return pmap_heuristic_train
    else:
        return jax.jit(heuristic_train)

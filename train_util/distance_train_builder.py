"""Shared distance-style neural trainer."""

from __future__ import annotations

import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.sampling import minibatch_datasets
from train_util.train_state import TrainStateExtended, hard_update_target, soft_update_target
from train_util.util import apply_with_conditional_batch_stats


def distance_train_builder(
    *,
    minibatch_size: int,
    model: DistanceModel | DistanceHLGModel,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    target_keys: tuple[str, ...],
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: dict[str, Any] | None = None,
    replay_ratio: int = 1,
    use_soft_update: bool = False,
    update_interval: int = 100,
    soft_update_tau: float = 0.005,
    enable_jit_hard_update: bool = True,
):
    """Build the shared train loop for heuristic and Q-function distance models."""

    def train_loss(
        params: Any,
        batch_stats: Any,
        solveconfigs: chex.Array,
        states: chex.Array,
        *loss_inputs_and_weight_key: Any,
    ):
        *loss_inputs, weights, key = loss_inputs_and_weight_key
        full_params = {"params": params}
        if batch_stats is not None:
            full_params["batch_stats"] = batch_stats

        preproc = jax.vmap(preproc_fn)(solveconfigs, states)
        (per_sample_loss, log_infos), variable_updates = apply_with_conditional_batch_stats(
            model.apply,
            full_params,
            preproc,
            *loss_inputs,
            training=True,
            n_devices=n_devices,
            method=model.train_loss,
            loss_type=loss_type,
            loss_args=loss_args,
            rngs={"params": key},
        )
        new_batch_stats = variable_updates.get("batch_stats", batch_stats)
        loss_value = jnp.mean(per_sample_loss.squeeze() * weights)
        return loss_value, (new_batch_stats, log_infos)

    def train(key: chex.PRNGKey, dataset: dict[str, chex.Array], state: TrainStateExtended):
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        targets = tuple(dataset[name] for name in target_keys)
        data_size = targets[0].shape[0]  # type: ignore[index]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            state, key = carry
            step_key, key = jax.random.split(key)
            solveconfigs_b, states_b, *target_batches, weights_b = batched_dataset

            (loss, (new_batch_stats, log_infos)), grads = jax.value_and_grad(
                train_loss, has_aux=True
            )(
                state.params,
                state.batch_stats,
                solveconfigs_b,
                states_b,
                *target_batches,
                weights_b,
                step_key,
            )

            if n_devices > 1:
                grads = jax.lax.psum(grads, axis_name="devices")

            updates, opt_state = optimizer.update(grads, state.opt_state, params=state.params)
            params = optax.apply_updates(state.params, updates)
            new_state = state.replace(
                params=params,
                batch_stats=new_batch_stats,
                opt_state=opt_state,
                step=state.step + 1,
            )

            if use_soft_update:
                new_state = soft_update_target(new_state, soft_update_tau)
            elif enable_jit_hard_update:
                should_update = (new_state.step % update_interval == 0) & (new_state.step > 0)
                new_state = jax.lax.cond(should_update, hard_update_target, lambda s: s, new_state)

            return (new_state, key), (loss, log_infos)

        def replay_loop(state, replay_key):
            key_replay, key_train = jax.random.split(replay_key)
            batched = minibatch_datasets(
                solveconfigs,  # type: ignore[arg-type]
                states,  # type: ignore[arg-type]
                *targets,  # type: ignore[arg-type]
                loss_weights,  # type: ignore[arg-type]
                data_size=data_size,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                key=key_replay,
            )  # type: ignore[arg-type]

            (state, _), (losses, log_infos) = jax.lax.scan(
                train_loop,
                (state, key_train),
                batched,
            )
            return state, (losses, log_infos)

        replay_keys = jax.random.split(key, replay_ratio)
        new_state, (losses, log_infos) = jax.lax.scan(replay_loop, state, replay_keys)
        return new_state, jnp.mean(losses), log_infos

    if n_devices > 1:

        def pmap_train(key, dataset, state):
            keys = jax.random.split(key, n_devices)
            new_state, loss, log_infos = jax.pmap(train, in_axes=(0, 0, None), axis_name="devices")(
                keys, dataset, state
            )
            return jax.tree_util.tree_map(lambda xs: xs[0], new_state), jnp.mean(loss), log_infos

        return pmap_train

    return jax.jit(train)

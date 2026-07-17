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
from train_util.util import build_distance_train_loss


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
    train_loss = build_distance_train_loss(
        model, preproc_fn, loss_type, loss_args, n_devices=n_devices
    )

    def train(key: chex.PRNGKey, dataset: dict[str, chex.Array], state: TrainStateExtended):
        solve_configs = dataset["solve_config"]
        states = dataset["state"]
        targets = tuple(dataset[name] for name in target_keys)
        data_size = targets[0].shape[0]  # type: ignore[index]
        batch_size = math.ceil(data_size / minibatch_size)

        def train_loop(carry, batched_dataset):
            state, key = carry
            step_key, key = jax.random.split(key)
            solve_configs_b, states_b, *target_batches = batched_dataset

            (loss, (new_batch_stats, log_infos)), grads = jax.value_and_grad(
                train_loss, has_aux=True
            )(
                state.params,
                state.batch_stats,
                solve_configs_b,
                states_b,
                *target_batches,
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
                solve_configs,  # type: ignore[arg-type]
                states,  # type: ignore[arg-type]
                *targets,  # type: ignore[arg-type]
                data_size=data_size,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                key=key_replay,
            )

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

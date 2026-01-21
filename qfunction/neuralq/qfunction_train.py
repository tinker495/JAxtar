"""
Q-function training builder using TrainStateExtended.

JAX/Flax 표준 패턴을 따라 학습 상태를 TrainStateExtended로 캡슐화합니다.
"""

import math
from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.sampling import minibatch_datasets
from train_util.train_state import (
    TrainStateExtended,
    hard_update_target,
    soft_update_target,
)
from train_util.util import (
    apply_with_conditional_batch_stats,
    get_self_predictive_train_args,
)


def qfunction_train_builder(
    minibatch_size: int,
    q_fn: DistanceModel | DistanceHLGModel,
    optimizer: optax.GradientTransformation,
    preproc_fn: Callable,
    n_devices: int = 1,
    loss_type: str = "mse",
    loss_args: Optional[dict[str, Any]] = None,
    replay_ratio: int = 1,
    # Target update options
    use_soft_update: bool = False,
    update_interval: int = 100,
    soft_update_tau: float = 0.005,
):
    """
    Build a Q-function training function that operates on TrainStateExtended.

    Args:
        minibatch_size: Size of each minibatch.
        q_fn: The Q-function model (DistanceModel or DistanceHLGModel).
        optimizer: The optax optimizer.
        preproc_fn: Preprocessing function for inputs.
        n_devices: Number of devices for pmap.
        loss_type: Type of loss function.
        loss_args: Additional loss function arguments.
        replay_ratio: Number of replay iterations per dataset.
        use_soft_update: Whether to use soft (Polyak) target update.
        update_interval: Steps between hard target updates (if not using soft update).
        soft_update_tau: Tau for soft update (ignored if use_soft_update is False).

    Returns:
        A training function that takes (key, dataset, state) and returns (new_state, loss, log_infos).
    """

    def qfunction_train_loss(
        params: Any,
        batch_stats: Any,
        target_params: Any,
        preproc: chex.Array,
        actions: chex.Array,
        target_qs: chex.Array,
        path_actions: chex.Array,
        ema_latents: chex.Array,
        same_trajectory_masks: chex.Array,
        weights: chex.Array,
        key: chex.PRNGKey,
    ):
        """Compute loss and gradients for a single minibatch."""
        # Combine params and batch_stats for apply
        full_params = {"params": params}
        if batch_stats is not None:
            full_params["batch_stats"] = batch_stats

        (per_sample_loss, log_infos), variable_updates = apply_with_conditional_batch_stats(
            q_fn.apply,
            full_params,
            preproc,
            target_qs,
            actions,
            training=True,
            n_devices=n_devices,
            method=q_fn.train_loss,
            loss_type=loss_type,
            loss_args=loss_args,
            path_actions=path_actions,
            ema_latents=ema_latents,
            same_trajectory_masks=same_trajectory_masks,
            rngs={"params": key},
        )
        new_batch_stats = variable_updates.get("batch_stats", batch_stats)
        loss_value = jnp.mean(per_sample_loss.squeeze() * weights)
        return loss_value, (new_batch_stats, log_infos)

    def qfunction_train(
        key: chex.PRNGKey,
        dataset: dict[str, chex.Array],
        state: TrainStateExtended,
    ):
        """
        Run one training epoch on the dataset.

        Args:
            key: PRNG key.
            dataset: Dictionary with 'solveconfigs', 'states', 'actions', 'target_q'.
            state: Current TrainStateExtended.

        Returns:
            Tuple of (new_state, loss, log_infos).
        """
        solveconfigs = dataset["solveconfigs"]
        states = dataset["states"]
        target_q = dataset["target_q"]
        actions = dataset["actions"]
        path_actions = dataset["path_actions"]
        trajectory_indices = dataset["trajectory_indices"]
        step_indices = dataset["step_indices"]
        data_size = target_q.shape[0]
        batch_size = math.ceil(data_size / minibatch_size)

        loss_weights = jnp.ones(data_size)
        loss_weights = loss_weights / jnp.mean(loss_weights)

        def train_loop(carry, batched_dataset):
            state, key = carry
            step_key, key = jax.random.split(key)
            (
                solveconfigs,
                states,
                target_q,
                actions,
                path_actions,
                trajectory_indices,
                step_indices,
                weights,
            ) = batched_dataset

            preprocessed_states = jax.vmap(jax.vmap(preproc_fn))(solveconfigs, states)

            # Get target params for self-predictive learning
            target_params = {"params": state.target_params}
            if state.batch_stats is not None:
                target_params["batch_stats"] = state.batch_stats

            (
                ema_next_state_latents,
                path_actions,
                same_trajectory_masks,
            ) = get_self_predictive_train_args(
                q_fn,
                target_params,
                preprocessed_states,
                path_actions,
                trajectory_indices,
                step_indices,
            )

            (loss, (new_batch_stats, log_infos)), grads = jax.value_and_grad(
                qfunction_train_loss, has_aux=True
            )(
                state.params,
                state.batch_stats,
                state.target_params,
                preprocessed_states,
                actions,
                target_q,
                path_actions,
                ema_next_state_latents,
                same_trajectory_masks,
                weights,
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
            else:
                should_update = (new_state.step % update_interval == 0) & (new_state.step > 0)
                new_state = jax.lax.cond(
                    should_update,
                    hard_update_target,
                    lambda s: s,
                    new_state,
                )

            return (new_state, key), (loss, log_infos)

        def replay_loop(state, replay_key):
            key_replay, key_train = jax.random.split(replay_key)
            (
                batched_solveconfigs,
                batched_states,
                batched_target_q,
                batched_actions,
                batched_path_actions,
                batched_trajectory_indices,
                batched_step_indices,
                batched_weights,
            ) = minibatch_datasets(
                solveconfigs,
                states,
                target_q,
                actions,
                path_actions,
                trajectory_indices,
                step_indices,
                loss_weights,
                data_size=data_size,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                key=key_replay,
            )  # type: ignore[arg-type]

            (state, _), (losses, log_infos) = jax.lax.scan(
                train_loop,
                (state, key_train),
                (
                    batched_solveconfigs,
                    batched_states,
                    batched_target_q,
                    batched_actions,
                    batched_path_actions,
                    batched_trajectory_indices,
                    batched_step_indices,
                    batched_weights,
                ),
            )
            return state, (losses, log_infos)

        # Replay loop
        replay_keys = jax.random.split(key, replay_ratio)
        new_state, (losses, log_infos) = jax.lax.scan(
            replay_loop,
            state,
            replay_keys,
        )
        loss = jnp.mean(losses)

        return new_state, loss, log_infos

    if n_devices > 1:

        def pmap_qfunction_train(key, dataset, state):
            keys = jax.random.split(key, n_devices)
            new_state, loss, log_infos = jax.pmap(
                qfunction_train, in_axes=(0, 0, None), axis_name="devices"
            )(keys, dataset, state)
            # Take first device's state (params should be synced)
            new_state = jax.tree_util.tree_map(lambda xs: xs[0], new_state)
            loss = jnp.mean(loss)
            return new_state, loss, log_infos

        return pmap_qfunction_train
    else:
        return jax.jit(qfunction_train)

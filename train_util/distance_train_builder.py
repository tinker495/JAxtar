"""Shared distance-style neural trainer."""

from __future__ import annotations

import math
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import optax

from neural_util.basemodel import DistanceHLGModel, DistanceModel
from train_util.optimizer import GRADIENT_CLIP_NORM, get_eval_params
from train_util.sampling import minibatch_datasets
from train_util.train_state import TrainStateExtended
from train_util.util import build_distance_train_loss


_METRIC_EPSILON = 1e-12


def _tree_rms(tree) -> chex.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    square_sum = sum(
        (jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves),
        start=jnp.array(0.0, dtype=jnp.float32),
    )
    element_count = sum(leaf.size for leaf in leaves)
    return jnp.sqrt(square_sum / max(element_count, 1))


def _tree_leaf_rms(tree) -> chex.Array:
    return jnp.stack(
        [
            jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))))
            for leaf in jax.tree_util.tree_leaves(tree)
        ]
    )


def _tree_nonfinite_count(tree) -> chex.Array:
    return sum(
        (jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)),
        start=jnp.array(0, dtype=jnp.int32),
    )


def _relative_tree_gap(tree, reference_tree) -> chex.Array:
    difference = jax.tree_util.tree_map(
        lambda value, reference: value - reference, tree, reference_tree
    )
    return _tree_rms(difference) / (_tree_rms(tree) + _METRIC_EPSILON)


def target_online_gap(state: TrainStateExtended) -> chex.Array:
    eval_params = get_eval_params(state.opt_state, state.params)
    return _relative_tree_gap(eval_params, state.target_params)


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

            aggregate_grad_norm = optax.tree.norm(grads)
            normalized_grads = jax.tree_util.tree_map(lambda grad: grad / n_devices, grads)
            normalized_grad_norm = optax.tree.norm(normalized_grads)
            parameter_rms_before = _tree_rms(state.params)
            parameter_leaf_rms_before = _tree_leaf_rms(state.params)
            updates, opt_state = optimizer.update(grads, state.opt_state, params=state.params)
            params = optax.apply_updates(state.params, updates)
            update_rms = _tree_rms(updates)
            update_leaf_rms = _tree_leaf_rms(updates)
            metrics = {
                "aggregate_grad_norm": aggregate_grad_norm,
                "normalized_grad_norm": normalized_grad_norm,
                "grad_leaf_rms": _tree_leaf_rms(normalized_grads),
                "update_rms": update_rms,
                "update_leaf_rms": update_leaf_rms,
                "relative_update": update_rms / (parameter_rms_before + _METRIC_EPSILON),
                "relative_update_leaf": update_leaf_rms
                / (parameter_leaf_rms_before + _METRIC_EPSILON),
                "was_clipped": aggregate_grad_norm > GRADIENT_CLIP_NORM,
                "nonfinite_count": (
                    _tree_nonfinite_count(grads)
                    + _tree_nonfinite_count(updates)
                    + _tree_nonfinite_count(params)
                    + (~jnp.isfinite(loss)).astype(jnp.int32)
                ),
            }
            new_state = state.replace(
                params=params,
                batch_stats=new_batch_stats,
                opt_state=opt_state,
                step=state.step + 1,
            )

            return (new_state, key), (loss, log_infos, metrics)

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

            (state, _), (losses, log_infos, metrics) = jax.lax.scan(
                train_loop,
                (state, key_train),
                batched,
            )
            return state, (losses, log_infos, metrics)

        replay_keys = jax.random.split(key, replay_ratio)
        new_state, (losses, log_infos, metrics) = jax.lax.scan(replay_loop, state, replay_keys)
        diagnostics = {
            "scalars": {
                "Model/Parameter RMS": _tree_rms(new_state.params),
                "Optimizer/Gradient Global Norm Mean": jnp.mean(metrics["normalized_grad_norm"]),
                "Optimizer/Gradient Global Norm Max": jnp.max(metrics["normalized_grad_norm"]),
                "Optimizer/Aggregated Gradient Global Norm Mean": jnp.mean(
                    metrics["aggregate_grad_norm"]
                ),
                "Optimizer/Aggregated Gradient Global Norm Max": jnp.max(
                    metrics["aggregate_grad_norm"]
                ),
                "Optimizer/Update RMS": jnp.mean(metrics["update_rms"]),
                "Optimizer/Update to Parameter Ratio": jnp.mean(metrics["relative_update"]),
                "Optimizer/Clip Fraction": jnp.mean(metrics["was_clipped"]),
                "Health/Nonfinite Count": jnp.max(metrics["nonfinite_count"]),
                "Metrics/TD Target Online Gap Before Update": target_online_gap(new_state),
            },
            "histograms": {
                "Model/Layer Parameter RMS": _tree_leaf_rms(new_state.params),
                "Optimizer/Layer Gradient RMS": metrics["grad_leaf_rms"].reshape(-1),
                "Optimizer/Layer Update RMS": metrics["update_leaf_rms"].reshape(-1),
                "Optimizer/Layer Update to Parameter Ratio": metrics[
                    "relative_update_leaf"
                ].reshape(-1),
            },
        }
        return new_state, jnp.mean(losses), log_infos, diagnostics

    if n_devices > 1:

        def pmap_train(key, dataset, state):
            keys = jax.random.split(key, n_devices)
            new_state, loss, log_infos, diagnostics = jax.pmap(
                train, in_axes=(0, 0, None), axis_name="devices"
            )(keys, dataset, state)
            return (
                jax.tree_util.tree_map(lambda xs: xs[0], new_state),
                jnp.mean(loss),
                log_infos,
                jax.tree_util.tree_map(lambda xs: xs[0], diagnostics),
            )

        return pmap_train

    return jax.jit(train)

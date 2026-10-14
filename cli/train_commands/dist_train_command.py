import os
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from config.pydantic_models import DistTrainOptions, PuzzleOptions
from helpers.config_printer import print_config
from helpers.jax_compile import compile_with_example
from helpers.logger import create_logger
from helpers.rich_progress import trange
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from heuristic.neuralheuristic.target_dataset_builder import (
    get_heuristic_dataset_builder,
)
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.target_dataset_builder import get_qfunction_dataset_builder
from train_util.distance_train_builder import distance_train_builder
from train_util.distance_train_builder import target_online_gap as _target_online_gap
from train_util.optimizer import get_learning_rate
from train_util.train_logs import TrainLogInfo
from train_util.train_state import hard_update_target, soft_update_target

from ..config_utils import enrich_config
from .train_session import (
    finalize_training,
    init_train_key,
    resolve_eval_context,
    run_training_eval,
    setup_train_state,
)

# Compile the host-side diagnostic without changing the training graph.
target_online_gap = jax.jit(_target_online_gap)


def _should_update_hard_target(
    *,
    iteration: int,
    last_update_iteration: int,
    update_interval: int,
    force_update_interval: int,
    loss,
    loss_threshold: float,
) -> bool:
    regular_update = iteration > 0 and iteration % update_interval == 0
    force_update = iteration - last_update_iteration >= force_update_interval
    return force_update or (regular_update and bool(loss <= loss_threshold))


def _effective_update_interval(
    update_interval: int,
    *,
    n_devices: int,
) -> int:
    return max(1, update_interval // n_devices)


def _soft_update_tau(update_interval: int, *, n_devices: int) -> float:
    return 1.0 - (1.0 - 1.0 / update_interval) ** n_devices


@jax.jit
def _distance_scalars(
    log_info_leaves: list[TrainLogInfo], target: jax.Array
) -> tuple[dict[str, jax.Array], tuple[jax.Array, ...]]:
    scalars = {
        "Metrics/Mean Target": jnp.mean(target),
        "Metrics/Target Std": jnp.std(target),
        "Metrics/Target P95": jnp.percentile(target, 95),
    }
    td_error = next((info.data for info in log_info_leaves if info.name == "Losses/diff"), None)
    if td_error is not None:
        scalars["Metrics/TD Error RMS"] = jnp.sqrt(jnp.mean(jnp.square(td_error)))
        scalars["Metrics/TD Error P95"] = jnp.percentile(jnp.abs(td_error), 95)
    prediction = next((info.data for info in log_info_leaves if info.name == "Metrics/pred"), None)
    if prediction is not None:
        scalars["Metrics/Prediction Std"] = jnp.std(prediction)
        scalars["Metrics/Prediction P95"] = jnp.percentile(prediction, 95)
    return scalars, tuple(info.mean for info in log_info_leaves if info.log_mean)


@jax.jit
def _pack_scalar_values(groups: tuple[tuple[jax.Array, ...], ...]) -> tuple[jax.Array, ...]:
    return tuple(jnp.stack(values) for values in groups)


def _get_scalar_values(
    values: tuple[
        jax.Array,
        jax.Array,
        dict[str, jax.Array],
        dict[str, jax.Array],
        tuple[jax.Array, ...],
    ],
) -> tuple[
    np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], tuple[np.ndarray, ...]
]:
    leaves, structure = jax.tree.flatten(values)
    groups: dict[np.dtype, list[int]] = {}
    for index, leaf in enumerate(leaves):
        if leaf.shape != ():
            raise ValueError(f"Expected a scalar at logging boundary, got {leaf.shape}")
        groups.setdefault(np.dtype(leaf.dtype), []).append(index)
    # Keep integer diagnostics exact and enqueue every dtype group before transferring.
    host_vectors = jax.device_get(
        _pack_scalar_values(tuple(tuple(leaves[index] for index in ids) for ids in groups.values()))
    )
    host_leaves: dict[int, np.ndarray] = {}
    for vector, indices in zip(host_vectors, groups.values(), strict=True):
        for position, index in enumerate(indices):
            host_leaves[index] = vector[position : position + 1].reshape(())
    return structure.unflatten([host_leaves[index] for index in range(len(leaves))])


def _run_distance_training(
    *,
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    search_model,
    search_model_name: str,
    model_config: Dict[str, Any],
    puzzle_name: str,
    puzzle_bundle,
    train_options: DistTrainOptions,
    k_max: int,
    target_keys: tuple[str, ...],
    dataset_builder: Callable,
    steps: int,
    gradient_updates_per_iteration: int,
    config_title: str,
    config_key: str,
    logger_suffix: str,
    progress_title: str,
    save_prefix: str,
    include_loss_in_progress: bool = False,
    dataset_extra_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    eval_options = train_options.eval_options
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    n_devices = jax.local_device_count() if train_options.multi_device else 1
    update_interval = _effective_update_interval(
        train_options.update_interval,
        n_devices=n_devices,
    )
    force_update_interval = _effective_update_interval(
        train_options.force_update_interval,
        n_devices=n_devices,
    )

    if n_devices > 1:
        steps = steps // n_devices

    total_gradient_updates = steps * gradient_updates_per_iteration * n_devices
    soft_update_tau = (
        _soft_update_tau(train_options.update_interval, n_devices=n_devices)
        if train_options.use_soft_update
        else 0.0
    )

    config = {
        "puzzle_options": puzzle_opts,
        search_model_name: search_model.__class__.__name__,
        f"{search_model_name}_metadata": getattr(search_model, "metadata", {}),
        "train_options": train_options,
        "eval_options": eval_options,
        config_key: model_config,
        "derived_parameters": {
            "training_loop_iterations": steps,
            "total_gradient_updates": total_gradient_updates,
            "configured_update_interval_outer_iterations": train_options.update_interval,
            "update_interval_outer_iterations": update_interval,
            "configured_force_update_interval_outer_iterations": (
                train_options.force_update_interval
            ),
            "force_update_interval_outer_iterations": force_update_interval,
            "soft_update_tau": soft_update_tau,
            "n_devices": n_devices,
        },
    }
    print_config(config_title, enrich_config(config))
    logger = create_logger(train_options.logger, f"{puzzle_name}-{logger_suffix}", config)
    key = init_train_key(train_options)
    key, _ = jax.random.split(key)

    if n_devices > 1:
        print(f"Training with {n_devices} devices")

    model, optimizer, state = setup_train_state(
        search_model,
        train_options,
        n_devices=n_devices,
        steps=steps,
        one_iter_size=train_options.dataset_batch_size // train_options.train_minibatch_size,
    )

    train_fn = distance_train_builder(
        minibatch_size=train_options.train_minibatch_size,
        model=model,
        optimizer=optimizer,
        preproc_fn=search_model.pre_process,
        target_keys=target_keys,
        n_devices=n_devices,
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
    )

    diffusion_warmup_steps = 0
    if train_options.label == "warmup_td":
        # max(1, ...) keeps a warmup phase even in tiny smoke runs
        diffusion_warmup_steps = max(1, int(steps * train_options.warmup_ratio))

    dataset_kwargs = {
        "n_devices": n_devices,
        "temperature": train_options.temperature,
        "label": train_options.label,
        "diffusion_warmup_steps": diffusion_warmup_steps,
        "non_backtracking_steps": train_options.sampling_non_backtracking_steps,
    }
    if dataset_extra_kwargs:
        dataset_kwargs.update(dataset_extra_kwargs)

    get_datasets = dataset_builder(
        puzzle,
        search_model.pre_process,
        model,
        train_options.dataset_batch_size,
        k_max,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        train_options.using_triangular_sampling,
        **dataset_kwargs,
    )

    print("warming up dataset + train function")
    dataset_key, train_key = jax.random.split(jax.random.PRNGKey(0))
    warmup_dataset = get_datasets(state, dataset_key, 0)
    compile_with_example(train_fn, train_key, warmup_dataset, state)

    eval_interval = steps
    if train_options.eval_count > 0:
        eval_interval = max(1, steps // train_options.eval_count)

    pbar = trange(steps)
    eval_ctx = dict(
        train_options=train_options,
        search_model=search_model,
        search_model_name=search_model_name,
        eval_puzzle=eval_puzzle,
        eval_puzzle_name=eval_puzzle_name,
        eval_puzzle_opts=eval_puzzle_opts,
        eval_kwargs=eval_kwargs,
        logger=logger,
        pbar=pbar,
    )
    last_update_iteration = 0

    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(state, subkey, i)
        target = dataset["distance"]
        state, loss, log_infos, diagnostics = train_fn(key, dataset, state)
        log_info_leaves = jax.tree_util.tree_leaves(
            log_infos, is_leaf=lambda value: isinstance(value, TrainLogInfo)
        )
        loss_value, lr, diagnostic_scalars, scalars, means = _get_scalar_values(
            (
                loss,
                get_learning_rate(state.opt_state),
                diagnostics["scalars"],
                *_distance_scalars(log_info_leaves, target),
            )
        )
        mean_log_infos = [info for info in log_info_leaves if info.log_mean]
        for info, value in zip(mean_log_infos, means, strict=True):
            logger.log_scalar(info.mean_name, float(value), i)
        for name, value in diagnostic_scalars.items():
            logger.log_scalar(name, float(value), i)
        for name, value in scalars.items():
            logger.log_scalar(name, float(value), i)

        discription_logs = {"lr": lr}
        if include_loss_in_progress:
            discription_logs["loss"] = float(loss_value)
        discription_logs.update(
            {
                "target": float(scalars["Metrics/Mean Target"]),
                **{
                    info.short_name: float(value)
                    for info, value in zip(mean_log_infos, means, strict=True)
                },
            }
        )
        pbar.set_description(desc=progress_title, desc_dict=discription_logs)

        logger.log_scalar("Metrics/Learning Rate", float(lr), i)
        if i % 100 == 0:
            for v in log_info_leaves:
                if v.log_histogram:
                    logger.log_histogram(v.histogram_name, v.data, i)
            for name, values in diagnostics["histograms"].items():
                logger.log_histogram(name, values, i)
            logger.log_histogram("Metrics/Target", target, i)

        target_updated = False
        if train_options.use_soft_update:
            state = soft_update_target(state, soft_update_tau)
            target_updated = True
        elif _should_update_hard_target(
            iteration=i,
            last_update_iteration=last_update_iteration,
            update_interval=update_interval,
            force_update_interval=force_update_interval,
            loss=loss,
            loss_threshold=train_options.loss_threshold,
        ):
            state = hard_update_target(state)
            if train_options.opt_state_reset:
                state = state.replace(opt_state=optimizer.init(state.params))
            last_update_iteration = i
            target_updated = True
        # Warm the diagnostic on iteration zero; unchanged targets reuse the existing gap.
        logger.log_scalar(
            "Metrics/TD Target Online Gap After Update",
            float(
                target_online_gap(state)
                if target_updated or i == 0
                else diagnostic_scalars["Metrics/TD Target Online Gap Before Update"]
            ),
            i,
        )

        if train_options.eval_count > 0 and i % eval_interval == 0 and i != 0:
            search_model.params = state.get_full_eval_params()
            backup_path = os.path.join(logger.log_dir, f"{save_prefix}_{i}.pkl")
            search_model.save_model(path=backup_path)
            if eval_options.num_eval > 0:
                run_training_eval(
                    step=i,
                    eval_options=eval_options.light_eval_options,
                    output_dir=Path(logger.log_dir) / "evaluation" / f"step_{i}",
                    **eval_ctx,
                    **kwargs,
                )

    finalize_training(
        state=state,
        save_prefix=save_prefix,
        steps=steps,
        eval_options=eval_options,
        **eval_ctx,
        **kwargs,
    )


def run_heuristic_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    puzzle_bundle,
    train_options: DistTrainOptions,
    k_max: int,
    heuristic_config: Dict[str, Any],
    **kwargs,
):
    steps = train_options.steps // train_options.replay_ratio
    _run_distance_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        search_model=heuristic,
        search_model_name="heuristic",
        model_config=heuristic_config,
        puzzle_name=puzzle_name,
        puzzle_bundle=puzzle_bundle,
        train_options=train_options,
        k_max=k_max,
        target_keys=("distance",),
        dataset_builder=get_heuristic_dataset_builder,
        steps=steps,
        gradient_updates_per_iteration=train_options.replay_ratio,
        config_title="Heuristic Training Configuration",
        config_key="heuristic_config",
        logger_suffix="dist-train",
        progress_title="Heuristic Training",
        save_prefix="heuristic",
        **kwargs,
    )


def run_qfunction_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    puzzle_bundle,
    train_options: DistTrainOptions,
    k_max: int,
    q_config: Dict[str, Any],
    **kwargs,
):
    steps = train_options.steps
    _run_distance_training(
        puzzle=puzzle,
        puzzle_opts=puzzle_opts,
        search_model=qfunction,
        search_model_name="qfunction",
        model_config=q_config,
        puzzle_name=puzzle_name,
        puzzle_bundle=puzzle_bundle,
        train_options=train_options,
        k_max=k_max,
        target_keys=("distance", "action"),
        dataset_builder=get_qfunction_dataset_builder,
        steps=steps,
        gradient_updates_per_iteration=1,
        config_title="Q-Function Training Configuration",
        config_key="q_config",
        logger_suffix="dist-q-train",
        progress_title="Q-Function Training",
        save_prefix="qfunction",
        include_loss_in_progress=True,
        dataset_extra_kwargs={"use_double_dqn": train_options.use_double_dqn},
        **kwargs,
    )

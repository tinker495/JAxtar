import os
from pathlib import Path
from typing import Any, Callable, Dict

import click
import jax
import jax.numpy as jnp
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
from train_util.optimizer import get_learning_rate
from train_util.train_state import hard_update_target, soft_update_target

from ..config_utils import enrich_config
from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
)
from .train_session import (
    finalize_training,
    init_train_key,
    log_train_leaves,
    resolve_eval_context,
    run_training_eval,
    setup_train_state,
)


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
        mean_target = jnp.mean(target)
        state, loss, log_infos = train_fn(key, dataset, state)
        lr = get_learning_rate(state.opt_state)
        log_info_leaves = log_train_leaves(logger, log_infos, i)

        if include_loss_in_progress:
            discription_logs = {
                "lr": lr,
                "loss": float(loss),
                "target": float(mean_target),
                **{v.short_name: float(v.mean) for v in log_info_leaves if v.log_mean},
            }
        else:
            discription_logs = {
                "lr": lr,
                "target": float(mean_target),
                **{v.short_name: float(v.mean) for v in log_info_leaves if v.log_mean},
            }
        pbar.set_description(desc=progress_title, desc_dict=discription_logs)

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Metrics/Mean Target", mean_target, i)
        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target, i)

        if train_options.use_soft_update:
            state = soft_update_target(state, soft_update_tau)
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


@click.command()
@dist_puzzle_options
@dist_train_options(preset_category="heuristic_train", default_preset="davi")
@dist_heuristic_options
def heuristic_train_command(
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


@click.command()
@dist_puzzle_options
@dist_train_options(preset_category="qfunction_train", default_preset="qlearning")
@dist_qfunction_options
def qfunction_train_command(
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

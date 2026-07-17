import os
from pathlib import Path
from typing import Any, Callable, Dict

import click
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from cli.evaluation_runner import run_evaluation_sweep
from config import benchmark_bundles, resolve_algorithm_for_component
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
from train_util.optimizer import get_learning_rate, setup_optimizer
from train_util.target_update import scaled_by_reset
from train_util.train_logs import TrainLogInfo
from train_util.train_state import TrainStateExtended

from ..config_utils import enrich_config
from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
)


def _resolve_eval_context(
    puzzle: Puzzle,
    puzzle_name: str,
    puzzle_opts: PuzzleOptions,
    puzzle_bundle,
):
    eval_benchmark = getattr(puzzle_bundle, "eval_benchmark", None) if puzzle_bundle else None
    if not eval_benchmark:
        return puzzle, puzzle_name, puzzle_opts, {}

    benchmark_bundle = benchmark_bundles.get(eval_benchmark)
    if benchmark_bundle is None:
        raise click.UsageError(
            f"Eval benchmark '{eval_benchmark}' is not registered for puzzle '{puzzle_name}'."
        )

    benchmark_args = dict(benchmark_bundle.benchmark_args or {})
    benchmark_instance = benchmark_bundle.benchmark(**benchmark_args)

    eval_kwargs = {
        "benchmark": benchmark_instance,
        "benchmark_name": eval_benchmark,
        "benchmark_bundle": benchmark_bundle,
        "benchmark_cli_options": {
            "sample_limit": None,
            "sample_ids": None,
        },
    }
    return (
        benchmark_instance.puzzle,
        eval_benchmark,
        PuzzleOptions(puzzle=eval_benchmark),
        eval_kwargs,
    )


def _resolve_eval_search_entry(
    *,
    train_options: DistTrainOptions,
    search_model_name: str,
) -> tuple[str, callable, dict]:
    metric = (train_options.eval_search_metric or "").strip()
    if search_model_name == "heuristic":
        metric = metric or "astar_d"
        expected_component = "heuristic"
    elif search_model_name == "qfunction":
        metric = metric or "qstar"
        expected_component = "qfunction"
    else:
        raise click.UsageError(f"Unknown search model name '{search_model_name}'.")

    try:
        resolution = resolve_algorithm_for_component(metric, expected_component)
    except KeyError as exc:
        raise click.UsageError(f"Invalid --eval-search-metric '{metric}'.") from exc
    except ValueError as exc:
        raise click.UsageError(
            f"Invalid --eval-search-metric '{metric}' for {search_model_name} training."
        ) from exc

    return resolution


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
    target_key: str,
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
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = _resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()

    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices

    total_gradient_updates = (
        steps * gradient_updates_per_iteration * n_devices
        if train_options.multi_device
        else steps * gradient_updates_per_iteration
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
            "update_interval_gradient_steps": update_interval,
            "reset_interval_gradient_steps": reset_interval,
            "n_devices": n_devices,
        },
    }
    print_config(config_title, enrich_config(config))
    logger = create_logger(train_options.logger, f"{puzzle_name}-{logger_suffix}", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, _ = jax.random.split(key)

    model = search_model.model
    model_params = search_model.params

    if train_options.multi_device and n_devices > 1:
        print(f"Training with {n_devices} devices")

    params = model_params.get("params", model_params)
    batch_stats = model_params.get("batch_stats", None)

    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )

    state = TrainStateExtended.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        target_params=params,
    ).replace(opt_state=opt_state)

    soft_update_tau = 1.0 / update_interval if train_options.use_soft_update else 0.0
    enable_jit_hard_update = (
        not train_options.use_soft_update and train_options.loss_threshold == float("inf")
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
        use_soft_update=train_options.use_soft_update,
        update_interval=update_interval,
        soft_update_tau=soft_update_tau,
        enable_jit_hard_update=enable_jit_hard_update,
    )

    dataset_kwargs = {
        "n_devices": n_devices,
        "temperature": train_options.temperature,
        "label": train_options.label,
        "use_diffusion_distance_warmup": train_options.use_diffusion_distance_warmup,
        "diffusion_distance_warmup_steps": train_options.diffusion_distance_warmup_steps,
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
    last_reset_time = 0
    last_update_step = -1

    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(state, subkey, i)
        target = dataset[target_key]
        mean_target = jnp.mean(target)
        state, loss, log_infos = train_fn(key, dataset, state)
        lr = get_learning_rate(state.opt_state)
        log_info_leaves = jax.tree_util.tree_leaves(
            log_infos, is_leaf=lambda x: isinstance(x, TrainLogInfo)
        )

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
        for v in log_info_leaves:
            if v.log_mean:
                logger.log_scalar(v.mean_name, v.mean, i)
            if v.log_histogram and i % 100 == 0:
                logger.log_histogram(v.histogram_name, v.data, i)
        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target, i)

        current_step = int(state.step)
        if train_options.use_soft_update:
            last_update_step = current_step
        elif not enable_jit_hard_update:
            is_regular_update_step = (current_step % update_interval == 0) and (current_step > 0)
            should_force_update = (
                current_step - last_update_step >= train_options.force_update_interval
            )
            should_regular_update = is_regular_update_step and (
                loss <= train_options.loss_threshold
            )
            if should_force_update or should_regular_update:
                state = state.update_target_params(state.get_full_eval_params()["params"])
                if train_options.opt_state_reset:
                    state = state.replace(opt_state=optimizer.init(state.params))
                last_update_step = current_step
        elif current_step > 0:
            last_update_step = (current_step // update_interval) * update_interval

        if (
            (current_step - last_reset_time >= reset_interval)
            and (current_step > 0)
            and (current_step < total_gradient_updates * 2 / 3)
            and (last_update_step >= last_reset_time)
        ):
            last_reset_time = current_step
            reset_params = {"params": state.params}
            if state.batch_stats is not None:
                reset_params["batch_stats"] = state.batch_stats
            reset_params = scaled_by_reset(reset_params, key, train_options.tau)
            state = state.replace(
                params=reset_params["params"],
                opt_state=optimizer.init(reset_params["params"]),
            )

        if train_options.eval_count > 0 and i % eval_interval == 0 and i != 0:
            search_model.params = state.get_full_eval_params()
            backup_path = os.path.join(logger.log_dir, f"{save_prefix}_{i}.pkl")
            search_model.save_model(path=backup_path)
            if eval_options.num_eval > 0:
                light_eval_options = eval_options.light_eval_options
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_entry(
                    train_options=train_options, search_model_name=search_model_name
                )
                with pbar.pause():
                    run_evaluation_sweep(
                        puzzle=eval_puzzle,
                        puzzle_name=eval_puzzle_name,
                        search_model=search_model,
                        search_model_name=search_model_name,
                        run_label=run_label,
                        search_builder_fn=search_builder_fn,
                        eval_options=light_eval_options,
                        puzzle_opts=eval_puzzle_opts,
                        output_dir=eval_run_dir,
                        logger=logger,
                        step=i,
                        **eval_builder_kwargs,
                        **eval_kwargs,
                        **kwargs,
                    )

    search_model.params = state.get_full_eval_params()
    backup_path = os.path.join(logger.log_dir, f"{save_prefix}_final.pkl")
    search_model.save_model(path=backup_path)
    logger.log_artifact(backup_path, f"{save_prefix}_final", "model")

    if eval_options.num_eval > 0 and train_options.eval_count > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_entry(
            train_options=train_options, search_model_name=search_model_name
        )
        with pbar.pause():
            run_evaluation_sweep(
                puzzle=eval_puzzle,
                puzzle_name=eval_puzzle_name,
                search_model=search_model,
                search_model_name=search_model_name,
                run_label=run_label,
                search_builder_fn=search_builder_fn,
                eval_options=eval_options,
                puzzle_opts=eval_puzzle_opts,
                output_dir=eval_run_dir,
                logger=logger,
                step=steps,
                **eval_builder_kwargs,
                **eval_kwargs,
                **kwargs,
            )

    logger.close()


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
        target_keys=("target_heuristic",),
        target_key="target_heuristic",
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
        target_keys=("target_q", "actions"),
        target_key="target_q",
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

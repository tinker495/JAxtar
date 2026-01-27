import os
from pathlib import Path
from typing import Any, Dict

import click
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from cli.evaluation_runner import run_evaluation_sweep
from config import benchmark_bundles
from config.pydantic_models import DistTrainOptions, PuzzleOptions
from helpers.config_printer import print_config
from helpers.logger import create_logger
from helpers.rich_progress import trange
from heuristic.neuralheuristic.heuristic_train import heuristic_train_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from heuristic.neuralheuristic.target_dataset_builder import (
    get_heuristic_dataset_builder,
)
from JAxtar.beamsearch.heuristic_beam import beam_builder
from JAxtar.beamsearch.q_beam import qbeam_builder
from JAxtar.stars.astar_d import astar_d_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qfunction_train import qfunction_train_builder
from qfunction.neuralq.target_dataset_builder import get_qfunction_dataset_builder
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


def _resolve_eval_search_components(
    *,
    train_options: DistTrainOptions,
    search_model_name: str,
) -> tuple[str, callable, dict]:
    """
    Resolve which evaluation search algorithm to use during training.

    Returns:
      - run_label (str)
      - search_builder_fn (callable)
      - extra_kwargs (dict): e.g. node_metric_label for beam-style searches
    """
    metric = (train_options.eval_search_metric or "").strip()
    extra_kwargs: dict = {}

    if search_model_name == "heuristic":
        # Default evaluation for heuristic training matches existing behavior (A* Deferred).
        metric = metric or "astar_d"
        if metric == "astar":
            # NOTE: astar_builder exists in eval commands; keep train default as astar_d unless requested.
            from JAxtar.stars.astar import astar_builder

            return "astar", astar_builder, extra_kwargs
        if metric == "astar_d":
            return "astar_d", astar_d_builder, extra_kwargs
        if metric == "bi_astar":
            from JAxtar.bi_stars.bi_astar import bi_astar_builder

            return "bi_astar", bi_astar_builder, extra_kwargs
        if metric == "bi_astar_d":
            from JAxtar.bi_stars.bi_astar_d import bi_astar_d_builder

            return "bi_astar_d", bi_astar_d_builder, extra_kwargs
        if metric == "beam":
            extra_kwargs["node_metric_label"] = "Beam Slots"
            return "beam", beam_builder, extra_kwargs
        raise click.UsageError(
            f"Invalid --eval-search-metric '{metric}' for heuristic training. "
            "Choose one of: astar, astar_d, bi_astar, bi_astar_d, beam."
        )

    if search_model_name == "qfunction":
        metric = metric or "qstar"
        if metric == "qstar":
            return "qstar", qstar_builder, extra_kwargs
        if metric == "bi_qstar":
            from JAxtar.bi_stars.bi_qstar import bi_qstar_builder

            return "bi_qstar", bi_qstar_builder, extra_kwargs
        if metric == "qbeam":
            extra_kwargs["node_metric_label"] = "Beam Slots"
            return "qbeam", qbeam_builder, extra_kwargs
        raise click.UsageError(
            f"Invalid --eval-search-metric '{metric}' for qfunction training. "
            "Choose one of: qstar, bi_qstar, qbeam."
        )

    raise click.UsageError(f"Unknown search model name '{search_model_name}'.")


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
    eval_options = train_options.eval_options
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = _resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    # Calculate derived parameters first for logging
    # steps is total training loop iterations (each does replay_ratio gradient updates)
    steps = train_options.steps // train_options.replay_ratio
    # Intervals are in gradient-step units (matching state.step counter)
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()

    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices

    config = {
        "puzzle_options": puzzle_opts,
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": getattr(heuristic, "metadata", {}),
        "train_options": train_options,
        "eval_options": eval_options,
        "heuristic_config": heuristic_config,
        "derived_parameters": {
            "training_loop_iterations": steps,
            "total_gradient_updates": (
                steps * train_options.replay_ratio * n_devices
                if train_options.multi_device
                else steps * train_options.replay_ratio
            ),
            "update_interval_gradient_steps": update_interval,
            "reset_interval_gradient_steps": reset_interval,
            "n_devices": n_devices,
        },
    }
    print_config("Heuristic Training Configuration", enrich_config(config))
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-train", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    heuristic_model = heuristic.model
    heuristic_params = heuristic.params

    if train_options.multi_device and n_devices > 1:
        print(f"Training with {n_devices} devices")

    # Extract params and batch_stats for TrainState
    params = heuristic_params.get("params", heuristic_params)
    batch_stats = heuristic_params.get("batch_stats", None)

    # Setup optimizer
    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )

    # Create TrainStateExtended
    state = TrainStateExtended.create(
        apply_fn=heuristic_model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        target_params=params,  # Initialize target with current params
    )
    # Replace opt_state with the one from setup_optimizer (has schedule info)
    state = state.replace(opt_state=opt_state)

    # Calculate soft update tau for use in train_builder
    # Intervals are in gradient-step units, matching state.step counter
    soft_update_tau = 1.0 / update_interval if train_options.use_soft_update else 0.0

    # Disable JIT hard updates when loss_threshold is used (handled externally)
    enable_jit_hard_update = (
        not train_options.use_soft_update and train_options.loss_threshold == float("inf")
    )

    # Build training function
    heuristic_train_fn = heuristic_train_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        heuristic.pre_process,
        n_devices=n_devices,
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
        use_soft_update=train_options.use_soft_update,
        update_interval=update_interval,
        soft_update_tau=soft_update_tau,
        enable_jit_hard_update=enable_jit_hard_update,
    )

    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_model,
        train_options.dataset_batch_size,
        k_max,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        train_options.using_triangular_sampling,
        n_devices=n_devices,
        temperature=train_options.temperature,
        use_diffusion_distance=train_options.use_diffusion_distance,
        use_diffusion_distance_mixture=train_options.use_diffusion_distance_mixture,
        use_diffusion_distance_warmup=train_options.use_diffusion_distance_warmup,
        diffusion_distance_warmup_steps=train_options.diffusion_distance_warmup_steps,
        non_backtracking_steps=train_options.sampling_non_backtracking_steps,
    )

    # Calculate eval interval safely
    eval_interval = steps
    if train_options.eval_count > 0:
        eval_interval = max(1, steps // train_options.eval_count)

    pbar = trange(steps)
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update

    for i in pbar:
        key, subkey = jax.random.split(key)

        # Get dataset using TrainStateExtended directly
        dataset = get_datasets(state, subkey, i)
        target_heuristic = dataset["target_heuristic"]
        mean_target_heuristic = jnp.mean(target_heuristic)

        # Train step
        state, loss, log_infos = heuristic_train_fn(key, dataset, state)

        lr = get_learning_rate(state.opt_state)

        # Cache tree_leaves to avoid duplicate calls
        log_info_leaves = jax.tree_util.tree_leaves(
            log_infos, is_leaf=lambda x: isinstance(x, TrainLogInfo)
        )

        discription_logs = {
            "lr": lr,
            "target": float(mean_target_heuristic),
            **{v.short_name: float(v.mean) for v in log_info_leaves if v.log_mean},
        }

        pbar.set_description(
            desc="Heuristic Training",
            desc_dict=discription_logs,
        )
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_heuristic, i)

        # Log metrics
        for v in log_info_leaves:
            if v.log_mean:
                logger.log_scalar(v.mean_name, v.mean, i)
            if v.log_histogram and i % 100 == 0:
                logger.log_histogram(v.histogram_name, v.data, i)

        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target_heuristic, i)

        # Handle target updates outside JIT when loss_threshold or force_update is used
        # (When enable_jit_hard_update=True, updates are handled inside JIT)
        if not train_options.use_soft_update and not enable_jit_hard_update:
            current_step = int(state.step)

            # Track when regular interval-based update would occur
            is_regular_update_step = (current_step % update_interval == 0) and (current_step > 0)

            # Update conditions:
            # 1. Force update interval exceeded, OR
            # 2. Regular update interval + loss below threshold
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

        # Reset logic (only if target was updated since last reset)
        current_step = int(state.step)
        total_gradient_steps = (
            steps * train_options.replay_ratio * (n_devices if train_options.multi_device else 1)
        )
        if (
            (current_step - last_reset_time >= reset_interval)
            and (current_step > 0)
            and (current_step < total_gradient_steps * 2 / 3)
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
            heuristic.params = state.get_full_eval_params()
            backup_path = os.path.join(logger.log_dir, f"heuristic_{i}.pkl")
            heuristic.save_model(path=backup_path)
            # Log model as artifact
            if eval_options.num_eval > 0:
                light_eval_options = eval_options.light_eval_options
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_components(
                    train_options=train_options, search_model_name="heuristic"
                )
                with pbar.pause():
                    run_evaluation_sweep(
                        puzzle=eval_puzzle,
                        puzzle_name=eval_puzzle_name,
                        search_model=heuristic,
                        search_model_name="heuristic",
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

    heuristic.params = state.get_full_eval_params()
    backup_path = os.path.join(logger.log_dir, "heuristic_final.pkl")
    heuristic.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "heuristic_final", "model")

    # Evaluation
    if eval_options.num_eval > 0 and train_options.eval_count > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_components(
            train_options=train_options, search_model_name="heuristic"
        )
        with pbar.pause():
            run_evaluation_sweep(
                puzzle=eval_puzzle,
                puzzle_name=eval_puzzle_name,
                search_model=heuristic,
                search_model_name="heuristic",
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
    eval_options = train_options.eval_options
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = _resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    # Calculate derived parameters first for logging
    # Note: For qfunction, steps is already total gradient updates (no division by replay_ratio)
    steps = train_options.steps
    # Intervals are in gradient-step units (matching state.step counter)
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()

    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices

    config = {
        "puzzle_options": puzzle_opts,
        "qfunction": qfunction.__class__.__name__,
        "qfunction_metadata": getattr(qfunction, "metadata", {}),
        "train_options": train_options,
        "eval_options": eval_options,
        "q_config": q_config,
        "derived_parameters": {
            "training_loop_iterations": steps,
            "total_gradient_updates": steps * n_devices if train_options.multi_device else steps,
            "update_interval_gradient_steps": update_interval,
            "reset_interval_gradient_steps": reset_interval,
            "n_devices": n_devices,
        },
    }
    print_config("Q-Function Training Configuration", enrich_config(config))
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-q-train", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    qfunc_model = qfunction.model
    qfunc_params = qfunction.params

    if train_options.multi_device and n_devices > 1:
        print(f"Training with {n_devices} devices")

    # Extract params and batch_stats for TrainState
    params = qfunc_params.get("params", qfunc_params)
    batch_stats = qfunc_params.get("batch_stats", None)

    # Setup optimizer
    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )

    # Create TrainStateExtended
    state = TrainStateExtended.create(
        apply_fn=qfunc_model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        target_params=params,  # Initialize target with current params
    )
    # Replace opt_state with the one from setup_optimizer (has schedule info)
    state = state.replace(opt_state=opt_state)

    # Calculate soft update tau for use in train_builder
    # Intervals are in gradient-step units, matching state.step counter
    soft_update_tau = 1.0 / update_interval if train_options.use_soft_update else 0.0

    # Disable JIT hard updates when loss_threshold is used (handled externally)
    enable_jit_hard_update = (
        not train_options.use_soft_update and train_options.loss_threshold == float("inf")
    )

    # Build training function
    qfunction_train_fn = qfunction_train_builder(
        train_options.train_minibatch_size,
        qfunc_model,
        optimizer,
        qfunction.pre_process,
        n_devices=n_devices,
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
        use_soft_update=train_options.use_soft_update,
        update_interval=update_interval,
        soft_update_tau=soft_update_tau,
        enable_jit_hard_update=enable_jit_hard_update,
    )

    get_datasets = get_qfunction_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_model,
        train_options.dataset_batch_size,
        k_max,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        train_options.using_triangular_sampling,
        n_devices=n_devices,
        temperature=train_options.temperature,
        use_double_dqn=train_options.use_double_dqn,
        use_diffusion_distance=train_options.use_diffusion_distance,
        use_diffusion_distance_mixture=train_options.use_diffusion_distance_mixture,
        use_diffusion_distance_warmup=train_options.use_diffusion_distance_warmup,
        diffusion_distance_warmup_steps=train_options.diffusion_distance_warmup_steps,
        non_backtracking_steps=train_options.sampling_non_backtracking_steps,
    )

    # Calculate eval interval safely
    eval_interval = steps
    if train_options.eval_count > 0:
        eval_interval = max(1, steps // train_options.eval_count)

    pbar = trange(steps)
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update

    for i in pbar:
        key, subkey = jax.random.split(key)

        # Get dataset using TrainStateExtended directly
        dataset = get_datasets(state, subkey, i)
        target_q = dataset["target_q"]
        mean_target_q = jnp.mean(target_q)

        # Train step
        state, loss, log_infos = qfunction_train_fn(key, dataset, state)

        lr = get_learning_rate(state.opt_state)

        # Cache tree_leaves to avoid duplicate calls
        log_info_leaves = jax.tree_util.tree_leaves(
            log_infos, is_leaf=lambda x: isinstance(x, TrainLogInfo)
        )

        discription_logs = {
            "lr": lr,
            "loss": float(loss),
            "target": float(mean_target_q),
            **{v.short_name: float(v.mean) for v in log_info_leaves if v.log_mean},
        }

        pbar.set_description(
            desc="Q-Function Training",
            desc_dict=discription_logs,
        )

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_q, i)

        # Log metrics
        for v in log_info_leaves:
            if v.log_mean:
                logger.log_scalar(v.mean_name, v.mean, i)
            if v.log_histogram and i % 100 == 0:
                logger.log_histogram(v.histogram_name, v.data, i)

        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target_q, i)

        # Handle target updates outside JIT when loss_threshold or force_update is used
        # (When enable_jit_hard_update=True, updates are handled inside JIT)
        if not train_options.use_soft_update and not enable_jit_hard_update:
            current_step = int(state.step)

            # Track when regular interval-based update would occur
            is_regular_update_step = (current_step % update_interval == 0) and (current_step > 0)

            # Update conditions:
            # 1. Force update interval exceeded, OR
            # 2. Regular update interval + loss below threshold
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

        # Reset logic (only if target was updated since last reset)
        current_step = int(state.step)
        total_gradient_steps = steps * (n_devices if train_options.multi_device else 1)
        if (
            (current_step - last_reset_time >= reset_interval)
            and (current_step > 0)
            and (current_step < total_gradient_steps * 2 / 3)
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
            qfunction.params = state.get_full_eval_params()
            backup_path = os.path.join(logger.log_dir, f"qfunction_{i}.pkl")
            qfunction.save_model(path=backup_path)
            # Log model as artifact
            if eval_options.num_eval > 0:
                light_eval_options = eval_options.light_eval_options
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_components(
                    train_options=train_options, search_model_name="qfunction"
                )
                with pbar.pause():
                    run_evaluation_sweep(
                        puzzle=eval_puzzle,
                        puzzle_name=eval_puzzle_name,
                        search_model=qfunction,
                        search_model_name="qfunction",
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

    qfunction.params = state.get_full_eval_params()
    backup_path = os.path.join(logger.log_dir, "qfunction_final.pkl")
    qfunction.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "qfunction_final", "model")

    # Evaluation
    if eval_options.num_eval > 0 and train_options.eval_count > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        run_label, search_builder_fn, eval_builder_kwargs = _resolve_eval_search_components(
            train_options=train_options, search_model_name="qfunction"
        )
        with pbar.pause():
            run_evaluation_sweep(
                puzzle=eval_puzzle,
                puzzle_name=eval_puzzle_name,
                search_model=qfunction,
                search_model_name="qfunction",
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

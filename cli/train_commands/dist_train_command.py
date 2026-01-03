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
from config.pydantic_models import DistTrainOptions, EvalOptions, PuzzleOptions
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
from train_util.optimizer import get_eval_params, get_learning_rate, setup_optimizer
from train_util.target_update import scaled_by_reset, soft_update

from ..config_utils import enrich_config
from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
    eval_options,
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
        if metric == "beam":
            extra_kwargs["node_metric_label"] = "Beam Slots"
            return "beam", beam_builder, extra_kwargs
        raise click.UsageError(
            f"Invalid --eval-search-metric '{metric}' for heuristic training. "
            "Choose one of: astar, astar_d, beam."
        )

    if search_model_name == "qfunction":
        metric = metric or "qstar"
        if metric == "qstar":
            return "qstar", qstar_builder, extra_kwargs
        if metric == "qbeam":
            extra_kwargs["node_metric_label"] = "Beam Slots"
            return "qbeam", qbeam_builder, extra_kwargs
        raise click.UsageError(
            f"Invalid --eval-search-metric '{metric}' for qfunction training. "
            "Choose one of: qstar, qbeam."
        )

    raise click.UsageError(f"Unknown search model name '{search_model_name}'.")


@click.command()
@dist_puzzle_options
@dist_train_options(preset_category="heuristic_train", default_preset="davi")
@dist_heuristic_options
@eval_options
def heuristic_train_command(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    puzzle_bundle,
    train_options: DistTrainOptions,
    k_max: int,
    eval_options: EvalOptions,
    heuristic_config: Dict[str, Any],
    **kwargs,
):
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = _resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    # Calculate derived parameters first for logging
    steps = train_options.steps // train_options.replay_ratio
    update_interval = train_options.update_interval // train_options.replay_ratio
    reset_interval = train_options.reset_interval // train_options.replay_ratio
    n_devices = jax.device_count()

    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        reset_interval = reset_interval // n_devices

    config = {
        "puzzle_options": puzzle_opts,
        "heuristic": heuristic.__class__.__name__,
        "heuristic_metadata": getattr(heuristic, "metadata", {}),
        "train_options": train_options,
        "eval_options": eval_options,
        "heuristic_config": heuristic_config,
        "derived_parameters": {
            "effective_steps": steps,
            "effective_update_interval": update_interval,
            "effective_reset_interval": reset_interval,
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
    target_heuristic_params = heuristic.params
    heuristic_params = target_heuristic_params

    if train_options.multi_device and n_devices > 1:
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        heuristic_params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    heuristic_train_fn = heuristic_train_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        heuristic.pre_process,
        n_devices=n_devices,
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
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

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update
    eval_params = get_eval_params(opt_state, heuristic_params)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, eval_params, subkey, i)
        target_heuristic = dataset["target_heuristic"]
        mean_target_heuristic = jnp.mean(target_heuristic)

        (
            heuristic_params,
            opt_state,
            loss,
            auxs,
        ) = heuristic_train_fn(key, dataset, heuristic_params, opt_state)
        eval_params = get_eval_params(opt_state, heuristic_params)
        lr = get_learning_rate(opt_state)
        pbar.set_description(
            desc="Heuristic Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "target_heuristic": float(mean_target_heuristic),
            },
        )
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_heuristic, i)

        # Log auxiliary metrics
        for k, v in auxs.items():
            mean_v = jnp.mean(v)
            logger.log_scalar(f"Aux/{k}", mean_v, i)
            if i % 100 == 0:
                logger.log_histogram(f"Aux/Dist_{k}", v, i)

        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target_heuristic, i)

        target_updated = False
        if train_options.use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, eval_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            if i % update_interval == 0 and i != 0:
                target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_heuristic_params = eval_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = optimizer.init(heuristic_params)
            target_updated = True
            last_update_step = i

        if (
            target_updated
            and i - last_reset_time >= reset_interval
            and updated
            and i < steps * 2 / 3
        ):
            last_reset_time = i
            heuristic_params = scaled_by_reset(
                heuristic_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(heuristic_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            heuristic.params = eval_params
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
    heuristic.params = eval_params
    backup_path = os.path.join(logger.log_dir, "heuristic_final.pkl")
    heuristic.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "heuristic_final", "model")

    # Evaluation
    if eval_options.num_eval > 0:
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
@eval_options
def qfunction_train_command(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    puzzle_bundle,
    train_options: DistTrainOptions,
    k_max: int,
    eval_options: EvalOptions,
    q_config: Dict[str, Any],
    **kwargs,
):
    eval_puzzle, eval_puzzle_name, eval_puzzle_opts, eval_kwargs = _resolve_eval_context(
        puzzle, puzzle_name, puzzle_opts, puzzle_bundle
    )
    # Calculate derived parameters first for logging
    steps = train_options.steps
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()

    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        reset_interval = reset_interval // n_devices

    config = {
        "puzzle_options": puzzle_opts,
        "qfunction": qfunction.__class__.__name__,
        "qfunction_metadata": getattr(qfunction, "metadata", {}),
        "train_options": train_options,
        "eval_options": eval_options,
        "q_config": q_config,
        "derived_parameters": {
            "effective_steps": steps,
            "effective_update_interval": update_interval,
            "effective_reset_interval": reset_interval,
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
    target_qfunc_params = qfunction.params
    qfunc_params = target_qfunc_params

    if train_options.multi_device and n_devices > 1:
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        qfunc_params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    qfunction_train_fn = qfunction_train_builder(
        train_options.train_minibatch_size,
        qfunc_model,
        optimizer,
        qfunction.pre_process,
        n_devices=n_devices,
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
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

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update
    eval_params = get_eval_params(opt_state, qfunc_params)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, eval_params, subkey, i)
        target_q = dataset["target_q"]
        mean_target_q = jnp.mean(target_q)

        (
            qfunc_params,
            opt_state,
            loss,
            auxs,
        ) = qfunction_train_fn(key, dataset, qfunc_params, opt_state)
        eval_params = get_eval_params(opt_state, qfunc_params)
        lr = get_learning_rate(opt_state)
        pbar.set_description(
            desc="Q-Function Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "target_q": float(mean_target_q),
            },
        )

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_q, i)

        # Log auxiliary metrics
        for k, v in auxs.items():
            mean_v = jnp.mean(v)
            logger.log_scalar(f"Aux/{k}", mean_v, i)
            if i % 100 == 0:
                logger.log_histogram(f"Aux/Dist_{k}", v, i)

        if i % 100 == 0:
            logger.log_histogram("Metrics/Target", target_q, i)

        target_updated = False
        if train_options.use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, eval_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            if i % update_interval == 0 and i != 0:
                target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_qfunc_params = eval_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = optimizer.init(qfunc_params)
            target_updated = True
            last_update_step = i

        if (
            target_updated
            and i - last_reset_time >= reset_interval
            and updated
            and i < steps * 2 / 3
        ):
            last_reset_time = i
            qfunc_params = scaled_by_reset(
                qfunc_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(qfunc_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            qfunction.params = eval_params
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
    qfunction.params = eval_params
    backup_path = os.path.join(logger.log_dir, "qfunction_final.pkl")
    qfunction.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "qfunction_final", "model")

    # Evaluation
    if eval_options.num_eval > 0:
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

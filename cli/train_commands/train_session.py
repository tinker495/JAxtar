"""Shared session plumbing for the distance-training commands.

Everything here is identical between the dataset-driven trainer
(`dist_train_command`) and the replay-driven WBS trainer
(`wbs_train_command`): eval resolution, optimizer/state setup, in-training
eval sweeps, TrainLogInfo emission and final checkpointing.
"""

import os
from pathlib import Path

import click
import jax
import numpy as np
from puxle import Puzzle

from cli.evaluation_runner import run_evaluation_sweep
from config import benchmark_bundles, resolve_algorithm_for_component
from config.pydantic_models import DistTrainOptions, PuzzleOptions
from train_util.optimizer import setup_optimizer
from train_util.train_logs import TrainLogInfo
from train_util.train_state import TrainStateExtended


def resolve_eval_context(
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


def resolve_eval_search_entry(
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


def init_train_key(train_options: DistTrainOptions) -> jax.Array:
    return jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )


def setup_train_state(
    search_model,
    train_options: DistTrainOptions,
    *,
    n_devices: int,
    steps: int,
    one_iter_size: int,
):
    """Split model params, build the optimizer and the initial TrainStateExtended."""
    model = search_model.model
    model_params = search_model.params
    params = model_params.get("params", model_params)
    batch_stats = model_params.get("batch_stats", None)

    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        one_iter_size,
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
    return model, optimizer, state


def run_training_eval(
    *,
    step: int,
    eval_options,
    output_dir: Path,
    train_options: DistTrainOptions,
    search_model,
    search_model_name: str,
    eval_puzzle: Puzzle,
    eval_puzzle_name: str,
    eval_puzzle_opts: PuzzleOptions,
    eval_kwargs: dict,
    logger,
    pbar,
    **kwargs,
):
    run_label, search_builder_fn, eval_builder_kwargs = resolve_eval_search_entry(
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
            output_dir=output_dir,
            logger=logger,
            step=step,
            **eval_builder_kwargs,
            **eval_kwargs,
            **kwargs,
        )


def log_train_leaves(logger, log_infos, step: int) -> list[TrainLogInfo]:
    """Emit TrainLogInfo scalars/histograms and return the leaves for progress display."""
    log_info_leaves = jax.tree_util.tree_leaves(
        log_infos, is_leaf=lambda x: isinstance(x, TrainLogInfo)
    )
    for v in log_info_leaves:
        if v.log_mean:
            logger.log_scalar(v.mean_name, v.mean, step)
        if v.log_histogram and step % 100 == 0:
            logger.log_histogram(v.histogram_name, v.data, step)
    return log_info_leaves


def finalize_training(
    *,
    state: TrainStateExtended,
    save_prefix: str,
    steps: int,
    eval_options,
    train_options: DistTrainOptions,
    search_model,
    search_model_name: str,
    eval_puzzle: Puzzle,
    eval_puzzle_name: str,
    eval_puzzle_opts: PuzzleOptions,
    eval_kwargs: dict,
    logger,
    pbar,
    **kwargs,
):
    """Save the final checkpoint, run the closing evaluation sweep and close the logger."""
    search_model.params = state.get_full_eval_params()
    backup_path = os.path.join(logger.log_dir, f"{save_prefix}_final.pkl")
    search_model.save_model(path=backup_path)
    logger.log_artifact(backup_path, f"{save_prefix}_final", "model")

    if train_options.eval_count > 0 and eval_options.num_eval > 0:
        run_training_eval(
            step=steps,
            eval_options=eval_options,
            output_dir=Path(logger.log_dir) / "evaluation",
            train_options=train_options,
            search_model=search_model,
            search_model_name=search_model_name,
            eval_puzzle=eval_puzzle,
            eval_puzzle_name=eval_puzzle_name,
            eval_puzzle_opts=eval_puzzle_opts,
            eval_kwargs=eval_kwargs,
            logger=logger,
            pbar=pbar,
            **kwargs,
        )

    logger.close()

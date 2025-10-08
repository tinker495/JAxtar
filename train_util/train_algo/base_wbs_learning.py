from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flashbax.utils import get_timestep_count
from puxle import Puzzle

from cli.eval_commands import _run_evaluation_sweep
from config.pydantic_models import EvalOptions, PuzzleOptions, WBSDistTrainOptions
from helpers.config_printer import print_config
from helpers.formatting import human_format
from helpers.logger import BaseLogger, create_logger
from helpers.replay import init_experience_replay
from helpers.rich_progress import trange
from train_util.optimizer import setup_optimizer


@dataclass
class ReplayMetrics:
    """Container for metrics derived from replay training samples."""

    pbar_items: Dict[str, float]
    scalars: Dict[str, float]
    histograms: Dict[str, jnp.ndarray]


BuildComponentsFn = Callable[
    [
        Any,  # buffer
        Any,  # optimizer
        int,  # batch_multiplier
    ],
    Tuple[
        Callable[
            [jax.random.PRNGKey, Any, Any, Any],
            Tuple[Any, Any, float, float, jnp.ndarray, jnp.ndarray, float, float],
        ],
        Callable[[Any, Any, jax.random.PRNGKey], Tuple[Any, int, int, jax.random.PRNGKey]],
    ],
]

MetricsFn = Callable[[jnp.ndarray], ReplayMetrics]


def run_wbs_learning(
    *,
    puzzle: Puzzle,
    puzzle_name: str,
    puzzle_opts: PuzzleOptions,
    model: Any,
    model_name: str,
    train_options: WBSDistTrainOptions,
    eval_options: EvalOptions,
    config: Dict[str, Any],
    log_title: str,
    log_name: str,
    progress_desc: str,
    checkpoint_prefix: str,
    build_components: BuildComponentsFn,
    metrics_fn: MetricsFn,
    eval_search_builder_fn: Callable[..., Any],
    eval_kwargs: Dict[str, Any],
    replay_use_action: bool = False,
    sample_interval: int = 10,
    save_interval: int = 100,
) -> None:
    """Shared WBS training loop for heuristic and Q-function variants."""

    print_config(log_title, config)
    logger: BaseLogger = create_logger(train_options.logger, log_name, config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    params = model.params

    n_devices = 1
    steps = train_options.steps
    if train_options.multi_device:
        n_devices = jax.device_count()
        steps = max(1, train_options.steps // n_devices)
        print(f"Training with {n_devices} devices")

    buffer, buffer_state = init_experience_replay(
        puzzle.SolveConfig.default(),
        puzzle.State.default(),
        max_length=train_options.replay_size,
        min_length=train_options.train_minibatch_size * 10,
        sample_batch_size=train_options.train_minibatch_size,
        add_batch_size=train_options.add_batch_size,
        use_action=replay_use_action,
    )

    batch_multiplier = max(
        1,
        (train_options.add_batch_size // max(1, train_options.train_minibatch_size))
        * max(1, train_options.replay_ratio),
    )
    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        batch_multiplier,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    replay_trainer, get_datasets = build_components(buffer, optimizer, batch_multiplier)

    eval_search_fn_cache: Dict[int, Callable] = {}
    light_eval_options = eval_options.light_eval_options if eval_options.num_eval > 0 else None
    eval_interval = max(1, steps // 5) if steps > 0 else 1

    pbar = trange(steps)
    pause_ctx = pbar.pause if hasattr(pbar, "pause") else lambda: nullcontext()

    for i in pbar:
        key, subkey = jax.random.split(key)
        if sample_interval > 0 and i % sample_interval == 0:
            t = time.time()
            buffer_state, search_count, solved_count, key = get_datasets(
                params, buffer_state, subkey
            )
            dt = time.time() - t
            if search_count > 0:
                solved_ratio = solved_count / search_count
            else:
                solved_ratio = 0.0
            logger.log_scalar("Samples/Data sample time", dt, i)
            logger.log_scalar("Samples/Search Count", search_count, i)
            logger.log_scalar("Samples/Solved Count", solved_count, i)
            logger.log_scalar("Samples/Solved Ratio", solved_ratio, i)

        (
            params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_targets,
            grad_magnitude,
            weight_magnitude,
        ) = replay_trainer(key, buffer_state, params, opt_state)

        lr = opt_state.hyperparams["learning_rate"]
        replay_size = get_timestep_count(buffer_state)
        metrics = metrics_fn(sampled_targets)

        desc_items = {
            "lr": lr,
            "loss": float(loss),
            "abs_diff": float(mean_abs_diff),
            "replay_size": f"{human_format(replay_size)}/{human_format(train_options.replay_size)}",
        }
        desc_items.update({k: float(v) for k, v in metrics.pbar_items.items()})
        pbar.set_description(desc=progress_desc, desc_dict=desc_items)

        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        for name, value in metrics.scalars.items():
            logger.log_scalar(name, value, i)
        logger.log_histogram("Losses/Diff", diffs, i)
        for name, values in metrics.histograms.items():
            logger.log_histogram(name, values, i)

        if save_interval > 0 and i % save_interval == 0 and i != 0:
            model.params = params
            backup_path = os.path.join(logger.log_dir, f"{checkpoint_prefix}_{i}.pkl")
            model.save_model(path=backup_path)

        if eval_options.num_eval > 0 and i % eval_interval == 0 and i != 0:
            model.params = params
            eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
            with pause_ctx():
                _run_evaluation_sweep(
                    puzzle=puzzle,
                    puzzle_name=puzzle_name,
                    search_model=model,
                    search_model_name=model_name,
                    search_builder_fn=eval_search_builder_fn,
                    eval_options=light_eval_options or eval_options,
                    puzzle_opts=puzzle_opts,
                    output_dir=eval_run_dir,
                    logger=logger,
                    step=i,
                    search_fn_cache=eval_search_fn_cache,
                    **eval_kwargs,
                )

    model.params = params
    backup_path = os.path.join(logger.log_dir, f"{checkpoint_prefix}_final.pkl")
    model.save_model(path=backup_path)
    logger.log_artifact(backup_path, f"{checkpoint_prefix}_final", "model")

    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        with pause_ctx():
            _run_evaluation_sweep(
                puzzle=puzzle,
                puzzle_name=puzzle_name,
                search_model=model,
                search_model_name=model_name,
                search_builder_fn=eval_search_builder_fn,
                eval_options=eval_options,
                puzzle_opts=puzzle_opts,
                output_dir=eval_run_dir,
                logger=logger,
                step=steps,
                search_fn_cache=eval_search_fn_cache,
                **eval_kwargs,
            )

    logger.close()

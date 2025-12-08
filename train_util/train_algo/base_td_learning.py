from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from cli.eval_commands import run_evaluation_sweep
from config.pydantic_models import DistTrainOptions, EvalOptions, PuzzleOptions
from helpers.config_printer import print_config
from helpers.logger import BaseLogger, create_logger
from helpers.rich_progress import trange
from train_util.optimizer import get_eval_params, get_learning_rate, setup_optimizer
from train_util.target_update import scaled_by_reset, soft_update


@dataclass
class DatasetMetrics:
    """Container for dataset-derived metrics logged during TD learning."""

    pbar_items: Dict[str, float]
    scalars: Dict[str, float]
    histograms: Dict[str, jnp.ndarray]


BuildComponentsFn = Callable[
    [
        Any,  # optimizer
        int,  # n_devices
    ],
    Tuple[
        Callable[
            [jax.random.PRNGKey, Dict[str, Any], Any, Any],
            Tuple[Any, Any, float, float, float, jnp.ndarray],
        ],
        Callable[[Any, Any, jax.random.PRNGKey], Dict[str, Any]],
    ],
]

MetricsFn = Callable[[Dict[str, Any]], DatasetMetrics]


def run_td_learning(
    *,
    puzzle: Puzzle,
    puzzle_name: str,
    puzzle_opts: PuzzleOptions,
    model: Any,
    model_name: str,
    train_options: DistTrainOptions,
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
    histogram_interval: int = 100,
) -> None:
    """Generic temporal-difference style training loop shared by DAVI/Q-learning."""

    print_config(log_title, config)
    logger: BaseLogger = create_logger(train_options.logger, log_name, config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    target_params = model.params
    params = target_params

    steps = train_options.steps // train_options.replay_ratio
    update_interval = train_options.update_interval // train_options.replay_ratio
    reset_interval = train_options.reset_interval // train_options.replay_ratio
    n_devices = jax.device_count()
    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        reset_interval = reset_interval // n_devices
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
        train_options.optimizer,
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    train_step, get_datasets = build_components(optimizer, n_devices)

    eval_search_fn_cache: Dict[int, Callable] = {}
    light_eval_options = eval_options.light_eval_options if eval_options.num_eval > 0 else None
    eval_interval = max(1, steps // 5) if steps > 0 else 1

    pbar = trange(steps)
    pause_ctx = pbar.pause if hasattr(pbar, "pause") else lambda: nullcontext()

    updated = False
    last_reset_time = 0
    last_update_step = -1
    eval_params = get_eval_params(opt_state, params)

    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_params, eval_params, subkey)
        dataset_metrics = metrics_fn(dataset)

        (
            params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
        ) = train_step(key, dataset, params, opt_state)
        eval_params = get_eval_params(opt_state, params)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        lr = get_learning_rate(opt_state)

        desc_items = {
            "lr": lr,
            "loss": float(loss),
            "abs_diff": float(mean_abs_diff),
        }
        desc_items.update({k: float(v) for k, v in dataset_metrics.pbar_items.items()})
        pbar.set_description(desc=progress_desc, desc_dict=desc_items)

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        for name, value in dataset_metrics.scalars.items():
            logger.log_scalar(name, value, i)
        if i % histogram_interval == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            for name, values in dataset_metrics.histograms.items():
                logger.log_histogram(name, values, i)

        target_updated = False
        if train_options.use_soft_update and update_interval > 0:
            target_params = soft_update(
                target_params, eval_params, float(1 - 1.0 / max(update_interval, 1))
            )
            updated = True
            if i % max(update_interval, 1) == 0 and i != 0:
                target_updated = True
        elif (
            (update_interval > 0 and i % update_interval == 0 and i != 0)
            and loss <= train_options.loss_threshold
        ) or (i - last_update_step >= train_options.force_update_interval):
            target_params = eval_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = optimizer.init(params)
            target_updated = True
            last_update_step = i

        if (
            target_updated
            and (reset_interval <= 0 or i - last_reset_time >= reset_interval)
            and updated
            and i < steps * 2 / 3
        ):
            last_reset_time = i
            params = scaled_by_reset(
                params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(params)
            updated = False

        if i % eval_interval == 0 and i != 0:
            model.params = eval_params
            backup_path = os.path.join(logger.log_dir, f"{checkpoint_prefix}_{i}.pkl")
            model.save_model(path=backup_path)
            if eval_options.num_eval > 0:
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                with pause_ctx():
                    run_evaluation_sweep(
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

    model.params = eval_params
    backup_path = os.path.join(logger.log_dir, f"{checkpoint_prefix}_final.pkl")
    model.save_model(path=backup_path)
    logger.log_artifact(backup_path, f"{checkpoint_prefix}_final", "model")

    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        with pause_ctx():
            run_evaluation_sweep(
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

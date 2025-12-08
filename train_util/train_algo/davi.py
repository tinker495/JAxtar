from typing import Any

import jax.numpy as jnp
from puxle import Puzzle

from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
)
from heuristic.neuralheuristic.davi import (
    get_davi_dataset_builder,
    regression_trainer_builder,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from JAxtar.stars.astar import astar_builder

from .base_td_learning import DatasetMetrics, run_td_learning


def run_davi_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    eval_options: EvalOptions,
    heuristic_config: NeuralCallableConfig,
    **kwargs: Any,
) -> None:
    """Execute distributed DAVI training loop."""
    config = {
        "puzzle_options": puzzle_opts,
        "heuristic_config": heuristic_config,
        "train_options": train_options,
        "eval_options": eval_options,
    }

    def build_components(optimizer, n_devices):
        trainer = regression_trainer_builder(
            train_options.train_minibatch_size,
            heuristic.model,
            optimizer,
            heuristic.pre_process,
            n_devices=n_devices,
            use_target_confidence_weighting=train_options.use_target_confidence_weighting,
            use_target_sharpness_weighting=train_options.use_target_sharpness_weighting,
            target_sharpness_alpha=train_options.target_sharpness_alpha,
            using_priority_sampling=train_options.using_priority_sampling,
            per_alpha=train_options.per_alpha,
            per_beta=train_options.per_beta,
            per_epsilon=train_options.per_epsilon,
            loss_type=train_options.loss,
            huber_delta=train_options.huber_delta,
            replay_ratio=train_options.replay_ratio,
            td_error_clip=train_options.td_error_clip,
        )
        datasets = get_davi_dataset_builder(
            puzzle,
            heuristic.pre_process,
            heuristic.model,
            train_options.dataset_batch_size,
            shuffle_length,
            train_options.dataset_minibatch_size,
            train_options.using_hindsight_target,
            train_options.using_triangular_sampling,
            n_devices=n_devices,
            temperature=train_options.temperature,
            td_error_clip=train_options.td_error_clip,
            use_diffusion_distance=train_options.use_diffusion_distance,
            use_diffusion_distance_mixture=train_options.use_diffusion_distance_mixture,
            use_diffusion_distance_warmup=train_options.use_diffusion_distance_warmup,
            diffusion_distance_warmup_steps=train_options.diffusion_distance_warmup_steps,
        )
        return trainer, datasets

    def extract_metrics(dataset):
        target_values = dataset["target_heuristic"]
        mean_target = float(jnp.mean(target_values))
        scalars = {"Metrics/Mean Target": mean_target}
        histograms = {"Metrics/Target": target_values}

        if "target_entropy" in dataset:
            target_entropy = dataset["target_entropy"]
            scalars["Metrics/Mean Target Entropy"] = float(jnp.mean(target_entropy))
            histograms["Metrics/Target Entropy"] = target_entropy

        return DatasetMetrics(
            pbar_items={"target_heuristic": mean_target},
            scalars=scalars,
            histograms=histograms,
        )

    run_td_learning(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        puzzle_opts=puzzle_opts,
        model=heuristic,
        model_name="heuristic",
        train_options=train_options,
        eval_options=eval_options,
        config=config,
        log_title="DAVI Training Configuration",
        log_name=f"{puzzle_name}-dist-train",
        progress_desc="DAVI Training",
        checkpoint_prefix="heuristic",
        build_components=build_components,
        metrics_fn=extract_metrics,
        eval_search_builder_fn=astar_builder,
        eval_kwargs=kwargs,
    )

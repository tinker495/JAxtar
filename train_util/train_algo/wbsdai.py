from typing import Any

import jax.numpy as jnp
from puxle import Puzzle

from config.pydantic_models import (
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
    WBSDistTrainOptions,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from heuristic.neuralheuristic.wbsdai import (
    regression_replay_trainer_builder,
    wbsdai_dataset_builder,
)
from JAxtar.astar import astar_builder

from .base_wbs_learning import ReplayMetrics, run_wbs_learning


def run_wbsdai_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: WBSDistTrainOptions,
    heuristic_config: NeuralCallableConfig,
    eval_options: EvalOptions,
    **kwargs: Any,
) -> None:
    """Execute distributed WBSD-AI training loop."""
    config = {
        "puzzle_options": puzzle_opts,
        "heuristic_config": heuristic_config,
        "train_options": train_options,
        "eval_options": eval_options,
    }

    def build_components(buffer, optimizer, batch_multiplier):
        trainer = regression_replay_trainer_builder(
            buffer,
            batch_multiplier,
            heuristic.pre_process,
            heuristic.model,
            optimizer,
        )
        datasets = wbsdai_dataset_builder(
            puzzle,
            heuristic,
            buffer,
            max_nodes=train_options.max_nodes,
            add_batch_size=train_options.add_batch_size,
            search_batch_size=train_options.search_batch_size,
            sample_ratio=train_options.sample_ratio,
            cost_weight=train_options.cost_weight,
            pop_ratio=train_options.pop_ratio,
            use_promising_branch=train_options.use_promising_branch,
        )
        return trainer, datasets

    def extract_metrics(sampled_targets):
        mean_target = float(jnp.mean(sampled_targets))
        scalars = {"Metrics/Mean Target": mean_target}
        histograms = {"Metrics/Target": sampled_targets}
        return ReplayMetrics(
            pbar_items={"target_heuristic": mean_target},
            scalars=scalars,
            histograms=histograms,
        )

    run_wbs_learning(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        puzzle_opts=puzzle_opts,
        model=heuristic,
        model_name="heuristic",
        train_options=train_options,
        eval_options=eval_options,
        config=config,
        log_title="WBSDAI Training Configuration",
        log_name=f"{puzzle_name}-{puzzle.size}-wbsdai",
        progress_desc="WBSDAI Training",
        checkpoint_prefix="heuristic",
        build_components=build_components,
        metrics_fn=extract_metrics,
        eval_search_builder_fn=astar_builder,
        eval_kwargs=kwargs,
        replay_use_action=False,
    )

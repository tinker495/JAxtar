from typing import Any

import jax.numpy as jnp
from puxle import Puzzle

from config.pydantic_models import (
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
    WBSDistTrainOptions,
)
from JAxtar.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.wbsdqi import (
    regression_replay_q_trainer_builder,
    wbsdqi_dataset_builder,
)

from .base_wbs_learning import ReplayMetrics, run_wbs_learning


def run_wbsdqi_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: WBSDistTrainOptions,
    q_config: NeuralCallableConfig,
    eval_options: EvalOptions,
    **kwargs: Any,
) -> None:
    """Execute distributed WBSD-QI training loop."""
    config = {
        "puzzle_options": puzzle_opts,
        "qfunction_config": q_config,
        "train_options": train_options,
        "eval_options": eval_options,
    }

    def build_components(buffer, optimizer, batch_multiplier):
        trainer = regression_replay_q_trainer_builder(
            buffer,
            batch_multiplier,
            qfunction.pre_process,
            qfunction.model,
            optimizer,
        )
        datasets = wbsdqi_dataset_builder(
            puzzle,
            qfunction,
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
            pbar_items={"target_q": mean_target},
            scalars=scalars,
            histograms=histograms,
        )

    run_wbs_learning(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        puzzle_opts=puzzle_opts,
        model=qfunction,
        model_name="qfunction",
        train_options=train_options,
        eval_options=eval_options,
        config=config,
        log_title="WBSDQI Training Configuration",
        log_name=f"{puzzle_name}-{puzzle.size}-wbsdqi",
        progress_desc="WBSDQI Training",
        checkpoint_prefix="qfunction",
        build_components=build_components,
        metrics_fn=extract_metrics,
        eval_search_builder_fn=qstar_builder,
        eval_kwargs=kwargs,
        replay_use_action=True,
    )

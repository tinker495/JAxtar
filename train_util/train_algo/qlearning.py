from typing import Any

import jax.numpy as jnp
from puxle import Puzzle

from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
)
from JAxtar.stars.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder

from .base_td_learning import DatasetMetrics, run_td_learning


def run_qlearning_training(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    with_policy: bool,
    eval_options: EvalOptions,
    q_config: NeuralCallableConfig,
    **kwargs: Any,
) -> None:
    """Execute distributed Q-learning training loop."""
    config = {
        "puzzle_options": puzzle_opts,
        "train_options": train_options,
        "eval_options": eval_options,
        "q_config": q_config,
    }

    def build_components(optimizer, n_devices):
        trainer = qlearning_builder(
            train_options.train_minibatch_size,
            qfunction.model,
            optimizer,
            qfunction.pre_process,
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
        datasets = get_qlearning_dataset_builder(
            puzzle,
            qfunction.pre_process,
            qfunction.model,
            train_options.dataset_batch_size,
            shuffle_length,
            train_options.dataset_minibatch_size,
            train_options.using_hindsight_target,
            train_options.using_triangular_sampling,
            n_devices=n_devices,
            with_policy=with_policy,
            temperature=train_options.temperature,
            td_error_clip=train_options.td_error_clip,
            use_double_dqn=train_options.use_double_dqn,
        )
        return trainer, datasets

    def extract_metrics(dataset):
        target_q = dataset["target_q"]
        mean_target = float(jnp.mean(target_q))
        scalars = {"Metrics/Mean Target": mean_target}
        histograms = {"Metrics/Target": target_q}

        pbar_items = {"target_q": mean_target}
        if "action_entropy" in dataset:
            action_entropy = dataset["action_entropy"]
            entropy_mean = float(jnp.mean(action_entropy))
            scalars["Metrics/Mean Action Entropy"] = entropy_mean
            histograms["Metrics/Action Entropy"] = action_entropy
            pbar_items["entropy"] = entropy_mean
        if "target_entropy" in dataset:
            target_entropy = dataset["target_entropy"]
            scalars["Metrics/Mean Target Entropy"] = float(jnp.mean(target_entropy))
            histograms["Metrics/Target Entropy"] = target_entropy

        return DatasetMetrics(
            pbar_items=pbar_items,
            scalars=scalars,
            histograms=histograms,
        )

    run_td_learning(
        puzzle=puzzle,
        puzzle_name=puzzle_name,
        puzzle_opts=puzzle_opts,
        model=qfunction,
        model_name="qfunction",
        train_options=train_options,
        eval_options=eval_options,
        config=config,
        log_title="Q-Learning Training Configuration",
        log_name=f"{puzzle_name}-dist-q-train",
        progress_desc="Q-Learning Training",
        checkpoint_prefix="qfunction",
        build_components=build_components,
        metrics_fn=extract_metrics,
        eval_search_builder_fn=qstar_builder,
        eval_kwargs=kwargs,
    )

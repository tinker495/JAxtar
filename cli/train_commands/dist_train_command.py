from datetime import datetime

import click
import jax
import jax.numpy as jnp
import numpy as np
import tensorboardX
from tqdm import trange

from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.optimizer import setup_optimizer
from neural_util.target_update import soft_update
from puzzle.puzzle_base import Puzzle
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder

from .dist_train_option import (
    heuristic_options,
    puzzle_options,
    qfunction_options,
    train_option,
)


def setup_logging(
    puzzle_name: str, puzzle_size: int, train_type: str
) -> tensorboardX.SummaryWriter:
    log_dir = (
        f"runs/{puzzle_name}_{puzzle_size}_{train_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    return tensorboardX.SummaryWriter(log_dir)


@click.command()
@puzzle_options
@heuristic_options
@train_option
def davi(
    puzzle: Puzzle,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    puzzle_size: int,
    steps: int,
    shuffle_length: int,
    dataset_batch_size: int,
    dataset_minibatch_size: int,
    train_minibatch_size: int,
    key: int,
    loss_threshold: float,
    update_interval: int,
    use_soft_update: bool,
    using_hindsight_target: bool,
    using_importance_sampling: bool,
    **kwargs,
):

    writer = setup_logging(puzzle_name, puzzle_size, "davi")
    heuristic_fn = heuristic.model.apply
    heuristic_params = heuristic.get_new_params()
    target_heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(
        heuristic_params, steps, dataset_batch_size // train_minibatch_size
    )
    davi_fn = davi_builder(train_minibatch_size, heuristic_fn, optimizer, using_importance_sampling)
    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_fn,
        dataset_batch_size,
        shuffle_length,
        dataset_minibatch_size,
        using_hindsight_target,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(heuristic_params, target_heuristic_params, subkey)
        target_heuristic = dataset[1]
        diffs = dataset[2]
        mean_target_heuristic = jnp.mean(target_heuristic)

        (
            heuristic_params,
            opt_state,
            loss,
            mean_abs_diff,
            _,
            grad_magnitude,
            weight_magnitude,
        ) = davi_fn(key, dataset, heuristic_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_heuristic: {float(mean_target_heuristic):.2f}"
        )
        if i % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, i)
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
            writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
            writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_heuristic, i)

        if use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, float(1 - 1.0 / update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_heuristic_params = heuristic_params

        if i % 1000 == 0 and i != 0:
            heuristic.params = target_heuristic_params
            heuristic.save_model(
                f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
            )
    heuristic.save_model(f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl")


@click.command()
@puzzle_options
@qfunction_options
@train_option
def qlearning(
    puzzle: Puzzle,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    puzzle_size: int,
    steps: int,
    shuffle_length: int,
    dataset_batch_size: int,
    dataset_minibatch_size: int,
    train_minibatch_size: int,
    key: int,
    loss_threshold: float,
    update_interval: int,
    use_soft_update: bool,
    using_hindsight_target: bool,
    using_importance_sampling: bool,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "qlearning")
    qfunc_fn = qfunction.model.apply
    qfunc_params = qfunction.get_new_params()
    target_qfunc_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(
        qfunc_params, steps, dataset_batch_size // train_minibatch_size
    )
    qlearning_fn = qlearning_builder(
        train_minibatch_size, qfunc_fn, optimizer, using_importance_sampling
    )
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_fn,
        dataset_batch_size,
        shuffle_length,
        dataset_minibatch_size,
        using_hindsight_target,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)
        target_q = dataset[1]
        diffs = dataset[3]
        mean_target_q = jnp.mean(target_q)

        (
            qfunc_params,
            opt_state,
            loss,
            mean_abs_diff,
            _,
            grad_magnitude,
            weight_magnitude,
        ) = qlearning_fn(key, dataset, qfunc_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_q: {float(mean_target_q):.2f}"
        )
        if i % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, i)
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_q, i)
            writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
            writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_q, i)

        if use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, qfunc_params, float(1 - 1.0 / update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_qfunc_params = qfunc_params

        if i % 1000 == 0 and i != 0:
            qfunction.params = target_qfunc_params
            qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")
    qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")

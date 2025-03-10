from datetime import datetime
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorboardX
from tqdm import trange

from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.puzzle_base import Puzzle
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder

from .dist_train_option import (
    heuristic_options,
    puzzle_options,
    qfunction_options,
    train_option,
)

PyTree = Any


def setup_logging(
    puzzle_name: str, puzzle_size: int, train_type: str
) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/{puzzle_name}_{puzzle_size}_{train_type}_repr_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree, steps: int) -> optax.OptState:
    lr_schedule = optax.polynomial_schedule(
        init_value=1e-3, end_value=1e-5, power=2.0, transition_steps=steps // 2
    )
    optimizer = optax.adam(lr_schedule)
    return optimizer, optimizer.init(params)


@jax.jit
def softupdate(params: PyTree, target_params: PyTree, tau: float) -> PyTree:
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)


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
    using_hindsight_target: bool,
    soft_update: bool,
    **kwargs,
):

    writer = setup_logging(puzzle_name, puzzle_size, "davi")
    heuristic_fn = heuristic.model.apply
    heuristic_params = heuristic.get_new_params()
    target_heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(
        heuristic_params, steps * dataset_batch_size // train_minibatch_size
    )
    davi_fn = davi_builder(train_minibatch_size, heuristic_fn, optimizer)
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
        dataset = get_datasets(target_heuristic_params, subkey)
        target_heuristic = dataset[1]
        mean_target_heuristic = jnp.mean(target_heuristic)

        heuristic_params, opt_state, loss, mean_abs_diff, diffs = davi_fn(
            key, dataset, heuristic_params, opt_state
        )
        pbar.set_description(
            f"loss: {loss:.4f}, mean_abs_diff: {mean_abs_diff:.2f}, mean_target_heuristic: {mean_target_heuristic:.4f}"
        )
        if i % 10 == 0:
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_heuristic, i)

        if soft_update:
            target_heuristic_params = softupdate(
                heuristic_params, target_heuristic_params, 1.0 / float(update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_heuristic_params = heuristic_params

        if i % 10000 == 0 and i != 0:
            heuristic.params = target_heuristic_params
            heuristic.save_model(
                f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}_repr.pkl"
            )

    heuristic.params = target_heuristic_params
    heuristic.save_model(
        f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}_repr.pkl"
    )


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
    using_hindsight_target: bool,
    soft_update: bool,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "qlearning")

    def qfunc_fn(params, solve_config, current):
        return qfunction.model.apply(
            params,
            solve_config,
            current,
            training=False,
            mutable=["batch_stats"],
            method=qfunction.model.distance,
        )

    def train_info_fn(params, solve_config, current, next_state):
        return qfunction.model.apply(
            params,
            solve_config,
            current,
            next_state,
            training=True,
            mutable=["batch_stats"],
            method=qfunction.model.train_info,
        )

    qfunc_params = qfunction.get_new_params()
    target_qfunc_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(
        qfunc_params, steps * dataset_batch_size // train_minibatch_size
    )
    qlearning_fn = qlearning_builder(train_minibatch_size, train_info_fn, optimizer)
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.solve_config_pre_process,
        qfunction.state_pre_process,
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
        target_q = dataset[3]
        mean_target_q = jnp.mean(target_q)

        (
            qfunc_params,
            opt_state,
            loss,
            mse_loss,
            similarity_loss,
            mean_abs_diff,
            diffs,
        ) = qlearning_fn(key, dataset, qfunc_params, opt_state)
        pbar.set_description(
            f"loss: {loss:.4f}(mse: {mse_loss:.4f}, similarity: {similarity_loss:.4f})"
            f", mean_abs_diff: {mean_abs_diff:.2f}, mean_target_q: {mean_target_q:.4f}"
        )
        if i % 10 == 0:
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/MSE", mse_loss, i)
            writer.add_scalar("Losses/Similarity", similarity_loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_q, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_q, i)

        if soft_update:
            target_qfunc_params = softupdate(
                qfunc_params, target_qfunc_params, 1.0 / float(update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_qfunc_params = qfunc_params

        if i % 10000 == 0 and i != 0:
            qfunction.params = target_qfunc_params
            qfunction.save_model(
                f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}_repr.pkl"
            )

    qfunction.params = target_qfunc_params
    qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}_repr.pkl")

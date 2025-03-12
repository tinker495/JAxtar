from collections import deque
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
    log_dir = (
        f"runs/{puzzle_name}_{puzzle_size}_{train_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree, steps: int) -> optax.OptState:
    lr_schedule = optax.polynomial_schedule(
        init_value=1e-3, end_value=1e-5, power=2.0, transition_steps=steps // 2
    )

    def adam(learning_rate):
        return optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(learning_rate),
        )

    optimizer = optax.inject_hyperparams(adam)(lr_schedule)
    return optimizer, optimizer.init(params)


class LossConvergeDetector:
    def __init__(self, window_size: int = 50, threshold: float = 1e-5):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = deque(maxlen=window_size)

    def update(self, loss: float) -> bool:
        """
        Update loss history and check if loss has converged.

        Args:
            loss: The current loss value

        Returns:
            bool: True if loss has converged, False otherwise
        """
        log_loss = np.log(loss)
        self.loss_history.append(log_loss)
        if len(self.loss_history) == self.window_size:
            # Use 2nd derivative to detect inflection point
            # Fit a 3rd degree polynomial to capture the curvature
            x = np.arange(self.window_size)
            coeffs = np.polyfit(x, list(self.loss_history), 2)

            # Calculate first derivative at the latest point
            # For a polynomial ax^2 + bx + c, the first derivative is 2ax + b
            a, b, c = coeffs
            x_latest = self.window_size - 1
            first_deriv = 2 * a * x_latest + b
            # If first derivative is greater than negative threshold, the loss is stabilizing or increasing
            return first_deriv > -self.threshold, first_deriv
        return False, 0


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

    loss_converge_detector = LossConvergeDetector(update_interval // 2)

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, subkey)
        target_heuristic = dataset[1]
        mean_target_heuristic = jnp.mean(target_heuristic)

        heuristic_params, opt_state, loss, mean_abs_diff, diffs = davi_fn(
            key, dataset, heuristic_params, opt_state
        )
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {loss:.4f}, abs_diff: {mean_abs_diff:.2f}"
            f", target_heuristic: {mean_target_heuristic:.2f}"
        )
        detect, first_deriv = loss_converge_detector.update(loss)

        if i % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, i)
            writer.add_scalar("Losses/Loss First Derivative", first_deriv, i)
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_heuristic, i)

        if (i % update_interval == 0 and i != 0) and detect:
            target_heuristic_params = heuristic_params

        if i % 1000 == 0 and i != 0:
            heuristic.params = target_heuristic_params
            heuristic.save_model(
                f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
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
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "qlearning")
    qfunc_fn = qfunction.model.apply
    qfunc_params = qfunction.get_new_params()
    target_qfunc_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(
        qfunc_params, steps * dataset_batch_size // train_minibatch_size
    )
    qlearning_fn = qlearning_builder(train_minibatch_size, qfunc_fn, optimizer)
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_fn,
        dataset_batch_size,
        shuffle_length,
        dataset_minibatch_size,
        using_hindsight_target,
    )

    loss_converge_detector = LossConvergeDetector(update_interval // 2)

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)
        target_heuristic = dataset[1]
        mean_target_heuristic = jnp.mean(target_heuristic)

        qfunc_params, opt_state, loss, mean_abs_diff, diffs = qlearning_fn(
            key, dataset, qfunc_params, opt_state
        )
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {loss:.4f}, abs_diff: {mean_abs_diff:.2f}"
            f", target_q: {mean_target_heuristic:.2f}"
        )
        detect, first_deriv = loss_converge_detector.update(loss)

        if i % 10 == 0:
            writer.add_scalar("Metrics/Learning Rate", lr, i)
            writer.add_scalar("Losses/Loss First Derivative", first_deriv, i)
            writer.add_scalar("Losses/Loss", loss, i)
            writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
            writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_heuristic, i)

        if (i % update_interval == 0 and i != 0) and detect:
            target_qfunc_params = qfunc_params

        if i % 1000 == 0 and i != 0:
            qfunction.params = target_qfunc_params
            qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")

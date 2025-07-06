from datetime import datetime

import click
import jax
import jax.numpy as jnp
import numpy as np
import tensorboardX
from puxle import Puzzle
from tqdm import trange

from config.pydantic_models import DistTrainOptions
from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.optimizer import setup_optimizer
from neural_util.target_update import scaled_by_reset, soft_update
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder

from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
)


def setup_logging(
    puzzle_name: str, puzzle_size: int, train_type: str
) -> tensorboardX.SummaryWriter:
    log_dir = (
        f"runs/{puzzle_name}_{puzzle_size}_{train_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    return tensorboardX.SummaryWriter(log_dir)


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_heuristic_options
def davi(
    puzzle: Puzzle,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    **kwargs,
):
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    writer = setup_logging(puzzle_name, puzzle.size, "davi")
    heuristic_model = heuristic.model
    target_heuristic_params = heuristic.params
    heuristic_params = scaled_by_reset(
        target_heuristic_params,
        key,
        train_options.tau,
    )

    steps = train_options.steps
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()
    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        reset_interval = reset_interval // n_devices
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        heuristic_params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
    )
    davi_fn = davi_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        train_options.using_importance_sampling,
        n_devices=n_devices,
    )
    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_model,
        train_options.dataset_batch_size,
        shuffle_length,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        n_devices=n_devices,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, heuristic_params, subkey)
        target_heuristic = dataset["target_heuristic"]
        diffs = dataset["diff"]
        mean_target_heuristic = jnp.mean(target_heuristic)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))

        (
            heuristic_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
        ) = davi_fn(key, dataset, heuristic_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_heuristic: {float(mean_target_heuristic):.2f}"
        )
        writer.add_scalar("Metrics/Learning Rate", lr, i)
        writer.add_scalar("Losses/Loss", loss, i)
        writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
        writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 10 == 0:
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_heuristic, i)

        if train_options.use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, float(1 - 1.0 / update_interval)
            )
            updated = True
        elif (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
            target_heuristic_params = heuristic_params
            updated = True

        if i - last_reset_time >= reset_interval and updated and i < steps / 3:
            last_reset_time = i
            heuristic_params = scaled_by_reset(
                heuristic_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(heuristic_params)
            updated = False

        if i % 1000 == 0 and i != 0:
            heuristic.params = heuristic_params
            heuristic.save_model()
    heuristic.params = heuristic_params
    heuristic.save_model()


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_qfunction_options
def qlearning(
    puzzle: Puzzle,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    with_policy: bool,
    **kwargs,
):
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    writer = setup_logging(puzzle_name, puzzle.size, "qlearning")
    qfunc_model = qfunction.model
    target_qfunc_params = qfunction.params
    qfunc_params = scaled_by_reset(
        target_qfunc_params,
        key,
        train_options.tau,
    )

    steps = train_options.steps
    update_interval = train_options.update_interval
    reset_interval = train_options.reset_interval
    n_devices = jax.device_count()
    if train_options.multi_device and n_devices > 1:
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        reset_interval = reset_interval // n_devices
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        qfunc_params,
        n_devices,
        steps,
        train_options.dataset_batch_size // train_options.train_minibatch_size,
    )
    qlearning_fn = qlearning_builder(
        train_options.train_minibatch_size,
        qfunc_model,
        optimizer,
        train_options.using_importance_sampling,
        n_devices=n_devices,
    )
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_model,
        train_options.dataset_batch_size,
        shuffle_length,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        n_devices=n_devices,
        with_policy=with_policy,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)
        target_q = dataset["target_q"]
        diffs = dataset["diff"]
        mean_target_q = jnp.mean(target_q)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))

        (
            qfunc_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
        ) = qlearning_fn(key, dataset, qfunc_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_q: {float(mean_target_q):.2f}"
        )

        writer.add_scalar("Metrics/Learning Rate", lr, i)
        writer.add_scalar("Losses/Loss", loss, i)
        writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        writer.add_scalar("Metrics/Mean Target", mean_target_q, i)
        writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 10 == 0:
            writer.add_histogram("Losses/Diff", diffs, i)
            writer.add_histogram("Metrics/Target", target_q, i)

        if train_options.use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, qfunc_params, float(1 - 1.0 / update_interval)
            )
            updated = True
        elif (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
            target_qfunc_params = qfunc_params
            updated = True

        if i - last_reset_time >= reset_interval and updated and i < steps / 3:
            last_reset_time = i
            qfunc_params = scaled_by_reset(
                qfunc_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(qfunc_params)
            updated = False

        if i % 1000 == 0 and i != 0:
            qfunction.params = qfunc_params
            qfunction.save_model()
    qfunction.params = qfunc_params
    qfunction.save_model()

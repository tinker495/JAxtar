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
from qfunction.zeroshotq.zeroshot_qlearning import (
    get_zeroshot_qlearning_dataset_builder,
    zeroshot_qlearning_builder,
)
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQFunctionBase

from .dist_train_option import (
    heuristic_options,
    puzzle_options,
    qfunction_options,
    train_option,
    zeroshot_qfunction_options,
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
    multi_device: bool,
    **kwargs,
):

    writer = setup_logging(puzzle_name, puzzle_size, "davi")
    heuristic_model = heuristic.model
    heuristic_params = heuristic.get_new_params()
    target_heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    n_devices = 1
    if multi_device:
        n_devices = jax.device_count()
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        heuristic_params, n_devices, steps, dataset_batch_size // train_minibatch_size
    )
    opt_state_init = opt_state
    davi_fn = davi_builder(
        train_minibatch_size,
        heuristic_model,
        optimizer,
        using_importance_sampling,
        n_devices=n_devices,
    )
    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_model,
        dataset_batch_size,
        shuffle_length,
        dataset_minibatch_size,
        using_hindsight_target,
        n_devices=n_devices,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, heuristic_params, subkey)
        target_heuristic = dataset[1]
        diffs = dataset[2]
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

        if use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, float(1 - 1.0 / update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_heuristic_params = heuristic_params
            opt_state = opt_state_init

        if i % 1000 == 0 and i != 0:
            heuristic.params = heuristic_params
            heuristic.save_model(
                f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
            )
    heuristic.params = heuristic_params
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
    multi_device: bool,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "qlearning")
    qfunc_model = qfunction.model
    qfunc_params = qfunction.get_new_params()
    target_qfunc_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    n_devices = 1
    if multi_device:
        n_devices = jax.device_count()
        steps = steps // n_devices
        update_interval = update_interval // n_devices
        print(f"Training with {n_devices} devices")

    optimizer, opt_state = setup_optimizer(
        qfunc_params, n_devices, steps, dataset_batch_size // train_minibatch_size
    )
    opt_state_init = opt_state
    qlearning_fn = qlearning_builder(
        train_minibatch_size, qfunc_model, optimizer, using_importance_sampling, n_devices=n_devices
    )
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_model,
        dataset_batch_size,
        shuffle_length,
        dataset_minibatch_size,
        using_hindsight_target,
        n_devices=n_devices,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)
        target_q = dataset[1]
        diffs = dataset[3]
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

        if use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, qfunc_params, float(1 - 1.0 / update_interval)
            )
        elif (i % update_interval == 0 and i != 0) and loss <= loss_threshold:
            target_qfunc_params = qfunc_params
            opt_state = opt_state_init
        if i % 1000 == 0 and i != 0:
            qfunction.params = qfunc_params
            qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")
    qfunction.params = qfunc_params
    qfunction.save_model(f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl")


@click.command()
@puzzle_options
@zeroshot_qfunction_options
@train_option
@click.option(
    "--lambda-reg", type=float, default=0.001, help="Coefficient for orthonormality regularization."
)
@click.option("--polyak-alpha", type=float, default=0.999, help="Coefficient for Polyak averaging.")
def zeroshot_qlearning(
    puzzle: Puzzle,
    zeroshot_qfunction: ZeroshotQFunctionBase,
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
    multi_device: bool,
    lambda_reg: float,
    polyak_alpha: float,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "zeroshot_qlearning")
    qfunc_model = zeroshot_qfunction.model
    goal_model = zeroshot_qfunction.goal_projector
    qfunc_params, goal_params = zeroshot_qfunction.get_new_params()
    target_qfunc_params = qfunc_params
    target_goal_params = goal_params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    n_devices = 1
    if multi_device:
        n_devices = jax.device_count()
        original_steps = steps
        steps = steps // n_devices
        print(
            f"Training with {n_devices} devices. Effective steps: {steps} per device (Total: {original_steps})"
        )

    steps_per_epoch = dataset_batch_size // train_minibatch_size
    optimizer_q, _ = setup_optimizer(qfunc_params, n_devices, steps, steps_per_epoch)
    optimizer_goal, _ = setup_optimizer(qfunc_params, n_devices, steps, steps_per_epoch)

    opt_state_q = optimizer_q.init(qfunc_params)
    opt_state_goal = optimizer_goal.init(qfunc_params)

    zqlearning_fn = zeroshot_qlearning_builder(
        minibatch_size=train_minibatch_size,
        goal_model=goal_model,
        zeroshotq_model=qfunc_model,
        optimizer_q=optimizer_q,
        optimizer_goal=optimizer_goal,
        lambda_reg=lambda_reg,
        polyak_alpha=polyak_alpha,
        n_devices=n_devices,
    )

    get_datasets = get_zeroshot_qlearning_dataset_builder(
        puzzle,
        zeroshot_qfunction.pre_process_solve_config,
        zeroshot_qfunction.pre_process_state,
        goal_model=goal_model,
        zeroshotq_model=qfunc_model,
        dataset_size=dataset_batch_size,
        shuffle_length=shuffle_length,
        dataset_minibatch_size=dataset_minibatch_size,
        using_hindsight_target=using_hindsight_target,
        n_devices=n_devices,
    )

    pbar = trange(steps)
    for i in pbar:
        key, data_key, train_key = jax.random.split(key, 3)
        dataset = get_datasets(q_params=qfunc_params, goal_params=goal_params, key=data_key)

        (
            qfunc_params,
            goal_params,
            target_qfunc_params,
            target_goal_params,
            _,
            _,
            opt_state_q,
            opt_state_goal,
            total_loss,
            mse_loss,
            reg_loss,
            grad_magnitude,
            weight_magnitude,
        ) = zqlearning_fn(
            key=train_key,
            dataset=dataset,
            q_params=qfunc_params,
            target_q_params=target_qfunc_params,
            goal_params=goal_params,
            target_goal_params=target_goal_params,
            opt_state_q=opt_state_q,
            opt_state_goal=opt_state_goal,
        )

        lr = opt_state_q.hyperparams["learning_rate"]

        pbar.set_description(
            f"lr: {lr:.4f}, total_loss: {float(total_loss):.4f}, mse: {float(mse_loss):.4f}, reg: {float(reg_loss):.4f}"
        )

        writer.add_scalar("Metrics/Learning Rate", lr, i)
        writer.add_scalar("Losses/Total Loss", total_loss, i)
        writer.add_scalar("Losses/MSE Loss", mse_loss, i)
        writer.add_scalar("Losses/Regularization Loss", reg_loss, i)
        writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)

        if i % 1000 == 0 and i != 0:
            zeroshot_qfunction.params = qfunc_params
            zeroshot_qfunction.save_model(
                f"qfunction/zeroshotq/model/params/{puzzle_name}_{puzzle_size}.pkl"
            )

    zeroshot_qfunction.params = qfunc_params
    zeroshot_qfunction.save_model(
        f"qfunction/zeroshotq/model/params/{puzzle_name}_{puzzle_size}.pkl"
    )

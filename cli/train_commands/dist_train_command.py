import click
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from config.pydantic_models import DistTrainOptions
from helpers.config_printer import print_config
from helpers.logger import TensorboardLogger
from helpers.rich_progress import trange
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
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "heuristic": heuristic.__class__.__name__,
        "train_options": train_options.dict(),
        "shuffle_length": shuffle_length,
        **kwargs,
    }
    print_config("DAVI Training Configuration", config)
    logger = TensorboardLogger(f"{puzzle_name}_{puzzle.size}_davi", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

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
        train_options.optimizer,
    )
    if train_options.opt_state_reset:
        inited_opt_state = opt_state
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
        train_options.using_triangular_sampling,
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
            desc="DAVI Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "abs_diff": float(mean_abs_diff),
                "target_heuristic": float(mean_target_heuristic),
            },
        )
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_heuristic, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_heuristic, i)

        if train_options.use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, float(1 - 1.0 / update_interval)
            )
            updated = True
        elif (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
            target_heuristic_params = heuristic_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = inited_opt_state

        if i - last_reset_time >= reset_interval and updated and i < steps * 2 / 3:
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
    logger.close()


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
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "qfunction": qfunction.__class__.__name__,
        "train_options": train_options.dict(),
        "shuffle_length": shuffle_length,
        "with_policy": with_policy,
        **kwargs,
    }
    print_config("Q-Learning Training Configuration", config)
    logger = TensorboardLogger(f"{puzzle_name}_{puzzle.size}_qlearning", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

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
        train_options.optimizer,
    )
    if train_options.opt_state_reset:
        inited_opt_state = opt_state
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
        train_options.using_triangular_sampling,
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
            desc="Q-Learning Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "abs_diff": float(mean_abs_diff),
                "target_q": float(mean_target_q),
            },
        )

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_q, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_q, i)

        if train_options.use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, qfunc_params, float(1 - 1.0 / update_interval)
            )
            updated = True
        elif (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
            target_qfunc_params = qfunc_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = inited_opt_state

        if i - last_reset_time >= reset_interval and updated and i < steps * 2 / 3:
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
    logger.close()

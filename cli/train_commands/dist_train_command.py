import time
from datetime import datetime

import click
import jax
import jax.numpy as jnp
import numpy as np
import tensorboardX
from puxle import Puzzle

from config.pydantic_models import DistTrainOptions
from helpers.replay import init_experience_replay
from helpers.rich_progress import trange
from heuristic.neuralheuristic.davi import (
    get_davi_dataset_builder,
    regression_trainer_builder,
)
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from heuristic.neuralheuristic.wbsdai import (
    regression_replay_trainer_builder,
    wbsdai_dataset_builder,
)
from neural_util.optimizer import setup_optimizer
from neural_util.target_update import scaled_by_reset, soft_update
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder
from qfunction.neuralq.wbsdqi import (
    regression_replay_q_trainer_builder,
    wbsdqi_dataset_builder,
)

from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_train_options,
    wbs_dist_train_options,
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
    davi_fn = regression_trainer_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        train_options.using_importance_sampling,
        n_devices=n_devices,
    )
    get_datasets = get_davi_dataset_builder(
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
            desc="DAVI Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "abs_diff": float(mean_abs_diff),
                "target_heuristic": float(mean_target_heuristic),
            },
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
            desc="Q-Learning Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "abs_diff": float(mean_abs_diff),
                "target_q": float(mean_target_q),
            },
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


@click.command()
@dist_puzzle_options
@dist_heuristic_options
@wbs_dist_train_options
def wbsdai(
    puzzle: Puzzle,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    puzzle_size: int,
    steps: int,
    replay_size: int,
    max_nodes: int,
    search_batch_size: int,
    add_batch_size: int,
    train_minibatch_size: int,
    sample_ratio: float,
    cost_weight: float,
    use_optimal_branch: bool,
    key: int,
    multi_device: bool,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "wbsdai")
    heuristic_model = heuristic.model
    heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    n_devices = 1
    if multi_device:
        n_devices = jax.device_count()
        steps = steps // n_devices
        print(f"Training with {n_devices} devices")

    buffer, buffer_state = init_experience_replay(
        puzzle.SolveConfig.default(),
        puzzle.State.default(),
        max_length=replay_size,
        min_length=train_minibatch_size * 10,
        sample_batch_size=train_minibatch_size,
        add_batch_size=add_batch_size,
    )
    optimizer, opt_state = setup_optimizer(
        heuristic_params,
        n_devices,
        steps,
        100,
    )
    replay_trainer = regression_replay_trainer_builder(
        buffer, 100, heuristic.pre_process, heuristic_model, optimizer
    )
    get_datasets = wbsdai_dataset_builder(
        puzzle,
        heuristic,
        buffer,
        max_nodes=max_nodes,
        add_batch_size=add_batch_size,
        search_batch_size=search_batch_size,
        sample_ratio=sample_ratio,
        cost_weight=cost_weight,
        use_optimal_branch=use_optimal_branch,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        if i % 10 == 0:
            t = time.time()
            buffer_state, search_count, solved_count, key = get_datasets(
                heuristic_params, buffer_state, subkey
            )
            dt = time.time() - t
            writer.add_scalar("Samples/Data sample time", dt, i)
            writer.add_scalar("Samples/Search Count", search_count, i)
            writer.add_scalar("Samples/Solved Count", solved_count, i)
            writer.add_scalar("Samples/Solved Ratio", solved_count / search_count, i)

        (
            heuristic_params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_target_heuristics,
            grad_magnitude,
            weight_magnitude,
        ) = replay_trainer(key, buffer_state, heuristic_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        mean_target_heuristic = jnp.mean(sampled_target_heuristics)
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_heuristic: {float(mean_target_heuristic):.2f}"
        )
        writer.add_scalar("Losses/Loss", loss, i)
        writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        writer.add_scalar("Metrics/Learning Rate", lr, i)
        writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        writer.add_scalar("Metrics/Mean Target", mean_target_heuristic, i)
        writer.add_histogram("Losses/Diff", diffs, i)
        writer.add_histogram("Metrics/Target", sampled_target_heuristics, i)

        if i % 100 == 0 and i != 0:
            heuristic.params = heuristic_params
            heuristic.save_model()

    heuristic.params = heuristic_params
    heuristic.save_model()


@click.command()
@dist_puzzle_options
@dist_qfunction_options
@wbs_dist_train_options
def wbsdqi(
    puzzle: Puzzle,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    puzzle_size: int,
    steps: int,
    replay_size: int,
    max_nodes: int,
    search_batch_size: int,
    add_batch_size: int,
    train_minibatch_size: int,
    sample_ratio: float,
    cost_weight: float,
    use_optimal_branch: bool,
    key: int,
    multi_device: bool,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size, "wbsdai")
    qfunction_model = qfunction.model
    qfunction_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    n_devices = 1
    if multi_device:
        n_devices = jax.device_count()
        steps = steps // n_devices
        print(f"Training with {n_devices} devices")

    buffer, buffer_state = init_experience_replay(
        puzzle.SolveConfig.default(),
        puzzle.State.default(),
        max_length=replay_size,
        min_length=train_minibatch_size * 10,
        sample_batch_size=train_minibatch_size,
        add_batch_size=add_batch_size,
        use_action=True,
    )
    optimizer, opt_state = setup_optimizer(
        qfunction_params,
        n_devices,
        steps,
        100,
    )
    replay_trainer = regression_replay_q_trainer_builder(
        buffer, 100, qfunction.pre_process, qfunction_model, optimizer
    )
    get_datasets = wbsdqi_dataset_builder(
        puzzle,
        qfunction,
        buffer,
        max_nodes=max_nodes,
        add_batch_size=add_batch_size,
        search_batch_size=search_batch_size,
        sample_ratio=sample_ratio,
        cost_weight=cost_weight,
        use_optimal_branch=use_optimal_branch,
    )

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        if i % 10 == 0:
            t = time.time()
            buffer_state, search_count, solved_count, key = get_datasets(
                qfunction_params, buffer_state, subkey
            )
            dt = time.time() - t
            writer.add_scalar("Samples/Data sample time", dt, i)
            writer.add_scalar("Samples/Search Count", search_count, i)
            writer.add_scalar("Samples/Solved Count", solved_count, i)
            writer.add_scalar("Samples/Solved Ratio", solved_count / search_count, i)

        (
            qfunction_params,
            opt_state,
            loss,
            mean_abs_diff,
            diffs,
            sampled_target_q,
            grad_magnitude,
            weight_magnitude,
        ) = replay_trainer(key, buffer_state, qfunction_params, opt_state)
        lr = opt_state.hyperparams["learning_rate"]
        mean_target_q = jnp.mean(sampled_target_q)
        pbar.set_description(
            f"lr: {lr:.4f}, loss: {float(loss):.4f}, abs_diff: {float(mean_abs_diff):.2f}"
            f", target_q: {float(mean_target_q):.2f}"
        )
        writer.add_scalar("Losses/Loss", loss, i)
        writer.add_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        writer.add_scalar("Metrics/Learning Rate", lr, i)
        writer.add_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        writer.add_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        writer.add_scalar("Metrics/Mean Target", mean_target_q, i)
        writer.add_histogram("Losses/Diff", diffs, i)
        writer.add_histogram("Metrics/Target", sampled_target_q, i)

        if i % 100 == 0 and i != 0:
            qfunction.params = qfunction_params
            qfunction.save_model()

    qfunction.params = qfunction_params
    qfunction.save_model()

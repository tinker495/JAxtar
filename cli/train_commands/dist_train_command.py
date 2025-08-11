import os
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from cli.eval_commands import _run_evaluation_sweep
from config.pydantic_models import (
    DistTrainOptions,
    EvalOptions,
    NeuralCallableConfig,
    PuzzleOptions,
)
from helpers.config_printer import print_config
from helpers.logger import create_logger
from helpers.rich_progress import trange
from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from heuristic.neuralheuristic.spr_davi import (
    get_spr_heuristic_dataset_builder,
    spr_davi_builder,
)
from heuristic.neuralheuristic.spr_neuralheuristic_base import SPRNeuralHeuristic
from JAxtar.astar import astar_builder
from JAxtar.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder
from qfunction.neuralq.spr_neuralq_base import SPRNeuralQFunction
from qfunction.neuralq.spr_qlearning import (
    get_spr_qlearning_dataset_builder,
    spr_qlearning_builder,
)
from train_util.optimizer import setup_optimizer
from train_util.target_update import scaled_by_reset, soft_update

from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
    dist_spr_heuristic_options,
    dist_spr_qfunction_options,
    dist_train_options,
    eval_options,
)


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_heuristic_options
@eval_options
def davi(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: NeuralHeuristicBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    eval_options: EvalOptions,
    heuristic_config: NeuralCallableConfig,
    **kwargs,
):

    config = {
        "puzzle_options": puzzle_opts,
        "heuristic_config": heuristic_config,
        "train_options": train_options,
        "eval_options": eval_options,
    }
    print_config("DAVI Training Configuration", config)
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-train", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    heuristic_model = heuristic.model
    target_heuristic_params = heuristic.params
    heuristic_params = target_heuristic_params

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
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    davi_fn = davi_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        heuristic.pre_process,
        n_devices=n_devices,
        use_target_confidence_weighting=train_options.use_target_confidence_weighting,
        using_priority_sampling=train_options.using_priority_sampling,
        per_alpha=train_options.per_alpha,
        per_beta=train_options.per_beta,
        per_epsilon=train_options.per_epsilon,
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
    last_update_step = -1  # Track last update step for force update
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, heuristic_params, subkey)
        target_heuristic = dataset["target_heuristic"]
        mean_target_heuristic = jnp.mean(target_heuristic)

        (
            heuristic_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
        ) = davi_fn(key, dataset, heuristic_params, opt_state)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
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

        target_updated = False
        if train_options.use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, heuristic_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            if i % update_interval == 0 and i != 0:
                target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_heuristic_params = heuristic_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = optimizer.init(heuristic_params)
            target_updated = True
            last_update_step = i

        if (
            target_updated
            and i - last_reset_time >= reset_interval
            and updated
            and i < steps * 2 / 3
        ):
            last_reset_time = i
            heuristic_params = scaled_by_reset(
                heuristic_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(heuristic_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            heuristic.params = heuristic_params
            backup_path = os.path.join(logger.log_dir, f"heuristic_{i}.pkl")
            heuristic.save_model(path=backup_path)
    heuristic.params = heuristic_params
    backup_path = os.path.join(logger.log_dir, "heuristic_final.pkl")
    heuristic.save_model(path=backup_path)

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        _run_evaluation_sweep(
            puzzle=puzzle,
            puzzle_name=puzzle_name,
            search_model=heuristic,
            search_model_name="heuristic",
            search_builder_fn=astar_builder,
            eval_options=eval_options,
            puzzle_opts=puzzle_opts,
            output_dir=eval_run_dir,
            logger=logger,
            step=steps,
            **kwargs,
        )

    logger.close()


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_qfunction_options
@eval_options
def qlearning(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: NeuralQFunctionBase,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    with_policy: bool,
    eval_options: EvalOptions,
    q_config: NeuralCallableConfig,
    **kwargs,
):

    config = {
        "puzzle_options": puzzle_opts,
        "train_options": train_options,
        "eval_options": eval_options,
        "q_config": q_config,
    }
    print_config("Q-Learning Training Configuration", config)
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-q-train", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    qfunc_model = qfunction.model
    target_qfunc_params = qfunction.params
    qfunc_params = target_qfunc_params

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
        lr_init=train_options.learning_rate,
        weight_decay_size=train_options.weight_decay_size,
    )
    qlearning_fn = qlearning_builder(
        train_options.train_minibatch_size,
        qfunc_model,
        optimizer,
        qfunction.pre_process,
        n_devices=n_devices,
        use_target_confidence_weighting=train_options.use_target_confidence_weighting,
        using_priority_sampling=train_options.using_priority_sampling,
        per_alpha=train_options.per_alpha,
        per_beta=train_options.per_beta,
        per_epsilon=train_options.per_epsilon,
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
        temperature=train_options.temperature,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)
        target_q = dataset["target_q"]
        mean_target_q = jnp.mean(target_q)

        (
            qfunc_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
        ) = qlearning_fn(key, dataset, qfunc_params, opt_state)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
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

        target_updated = False
        if train_options.use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, qfunc_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_qfunc_params = qfunc_params
            updated = True
            if train_options.opt_state_reset:
                opt_state = optimizer.init(qfunc_params)
            target_updated = True
            last_update_step = i

        if (
            target_updated
            and i - last_reset_time >= reset_interval
            and updated
            and i < steps * 2 / 3
        ):
            last_reset_time = i
            qfunc_params = scaled_by_reset(
                qfunc_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(qfunc_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            qfunction.params = qfunc_params
            backup_path = os.path.join(logger.log_dir, f"qfunction_{i}.pkl")
            qfunction.save_model(path=backup_path)
    qfunction.params = qfunc_params
    backup_path = os.path.join(logger.log_dir, "qfunction_final.pkl")
    qfunction.save_model(path=backup_path)

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        _run_evaluation_sweep(
            puzzle=puzzle,
            puzzle_name=puzzle_name,
            search_model=qfunction,
            search_model_name="qfunction",
            search_builder_fn=qstar_builder,
            eval_options=eval_options,
            puzzle_opts=puzzle_opts,
            output_dir=eval_run_dir,
            logger=logger,
            step=steps,
            **kwargs,
        )

    logger.close()


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_spr_heuristic_options
@eval_options
def spr_davi(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    heuristic: SPRNeuralHeuristic,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    eval_options: EvalOptions,
    heuristic_config: NeuralCallableConfig,
    **kwargs,
):
    config = {
        "puzzle_options": puzzle_opts,
        "heuristic_config": heuristic_config,
        "train_options": train_options,
        "eval_options": eval_options,
    }
    print_config("SPR-DAVI Training Configuration", config)
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-spr-davi", config)
    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if train_options.key == 0 else train_options.key
    )
    key, subkey = jax.random.split(key)

    heuristic_model = heuristic.model
    target_heuristic_params = heuristic.params
    # Initialize online network with slight noise from target network
    heuristic_params = scaled_by_reset(
        target_heuristic_params,
        key,
        train_options.tau,
    )

    steps = train_options.steps
    update_interval = (
        train_options.update_interval
        * train_options.dataset_batch_size
        // (train_options.train_minibatch_size * 2)
    )
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
    spr_davi_fn = spr_davi_builder(
        train_options.train_minibatch_size,
        heuristic_model,
        optimizer,
        heuristic.pre_process,
        spr_loss_weight=kwargs.get("spr_loss_weight", 1.0),
        ema_tau=(1 - 1 / update_interval),
        n_devices=n_devices,
        use_target_confidence_weighting=train_options.use_target_confidence_weighting,
    )
    get_datasets = get_spr_heuristic_dataset_builder(
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
        # The dataset builder now needs both online and target params to calculate the initial diff
        dataset = get_datasets(target_heuristic_params, heuristic_params, subkey)

        target_heuristic = dataset["target_heuristic"]
        mean_target_heuristic = jnp.mean(target_heuristic)

        (
            heuristic_params,
            target_heuristic_params,
            opt_state,
            loss,
            davi_loss,
            spr_loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
        ) = spr_davi_fn(key, dataset, heuristic_params, target_heuristic_params, opt_state)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))

        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            desc="SPR-DAVI Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "davi_loss": float(davi_loss),
                "spr_loss": float(spr_loss),
                "abs_diff": float(mean_abs_diff),
                "target_h": float(mean_target_heuristic),
            },
        )
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Total Loss", loss, i)
        logger.log_scalar("Losses/DAVI Loss", davi_loss, i)
        logger.log_scalar("Losses/SPR Loss", spr_loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_heuristic, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_heuristic, i)

        if not train_options.use_soft_update:
            if (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
                target_heuristic_params = heuristic_params
                updated = True
                if train_options.opt_state_reset:
                    opt_state = optimizer.init(heuristic_params)
        else:
            updated = True

        if i - last_reset_time >= reset_interval and updated and i < steps * 2 / 3:
            last_reset_time = i
            heuristic_params = scaled_by_reset(
                heuristic_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(heuristic_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            heuristic.params = heuristic_params
            backup_path = os.path.join(logger.log_dir, f"heuristic_{i}.pkl")
            heuristic.save_model(path=backup_path)
    heuristic.params = heuristic_params
    backup_path = os.path.join(logger.log_dir, "heuristic_final.pkl")
    heuristic.save_model(path=backup_path)

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        _run_evaluation_sweep(
            puzzle=puzzle,
            puzzle_name=puzzle_name,
            search_model=heuristic,
            search_model_name="heuristic",
            search_builder_fn=astar_builder,
            eval_options=eval_options,
            puzzle_opts=puzzle_opts,
            output_dir=eval_run_dir,
            logger=logger,
            step=steps,
            **kwargs,
        )

    logger.close()


@click.command()
@dist_puzzle_options
@dist_train_options
@dist_spr_qfunction_options
@eval_options
def spr_qlearning(
    puzzle: Puzzle,
    puzzle_opts: PuzzleOptions,
    qfunction: SPRNeuralQFunction,
    puzzle_name: str,
    train_options: DistTrainOptions,
    shuffle_length: int,
    with_policy: bool,
    eval_options: EvalOptions,
    q_config: NeuralCallableConfig,
    **kwargs,
):
    config = {
        "puzzle_options": puzzle_opts,
        "train_options": train_options,
        "eval_options": eval_options,
        "q_config": q_config,
    }
    print_config("SPR-Q-Learning Training Configuration", config)
    logger = create_logger(train_options.logger, f"{puzzle_name}-dist-spr-qlearning", config)
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
    update_interval = (
        train_options.update_interval
        * train_options.dataset_batch_size
        // (train_options.train_minibatch_size * 2)
    )
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
    spr_qlearning_fn = spr_qlearning_builder(
        train_options.train_minibatch_size,
        qfunc_model,
        optimizer,
        qfunction.pre_process,
        spr_loss_weight=kwargs.get("spr_loss_weight", 1.0),
        ema_tau=(1 - 1 / update_interval),
        n_devices=n_devices,
        use_target_confidence_weighting=train_options.use_target_confidence_weighting,
    )
    get_datasets = get_spr_qlearning_dataset_builder(
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
        temperature=train_options.temperature,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, qfunc_params, subkey)

        target_q = dataset["target_q"]
        mean_target_q = jnp.mean(target_q)

        (
            qfunc_params,
            target_qfunc_params,
            opt_state,
            loss,
            q_loss,
            spr_loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
        ) = spr_qlearning_fn(key, dataset, qfunc_params, target_qfunc_params, opt_state)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))

        lr = opt_state.hyperparams["learning_rate"]
        pbar.set_description(
            desc="SPR-Q-Learning Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "q_loss": float(q_loss),
                "spr_loss": float(spr_loss),
                "abs_diff": float(mean_abs_diff),
                "target_q": float(mean_target_q),
            },
        )
        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Total Loss", loss, i)
        logger.log_scalar("Losses/Q Loss", q_loss, i)
        logger.log_scalar("Losses/SPR Loss", spr_loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_q, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_q, i)

        if not train_options.use_soft_update:
            if (i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold:
                target_qfunc_params = qfunc_params
                updated = True
                if train_options.opt_state_reset:
                    opt_state = optimizer.init(qfunc_params)
        else:
            updated = True

        if i - last_reset_time >= reset_interval and updated and i < steps * 2 / 3:
            last_reset_time = i
            qfunc_params = scaled_by_reset(
                qfunc_params,
                key,
                train_options.tau,
            )
            opt_state = optimizer.init(qfunc_params)
            updated = False

        if i % (steps // 5) == 0 and i != 0:
            qfunction.params = qfunc_params
            backup_path = os.path.join(logger.log_dir, f"qfunction_{i}.pkl")
            qfunction.save_model(path=backup_path)
    qfunction.params = qfunc_params
    backup_path = os.path.join(logger.log_dir, "qfunction_final.pkl")
    qfunction.save_model(path=backup_path)

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        _run_evaluation_sweep(
            puzzle=puzzle,
            puzzle_name=puzzle_name,
            search_model=qfunction,
            search_model_name="qfunction",
            search_builder_fn=qstar_builder,
            eval_options=eval_options,
            puzzle_opts=puzzle_opts,
            output_dir=eval_run_dir,
            logger=logger,
            step=steps,
            **kwargs,
        )

    logger.close()

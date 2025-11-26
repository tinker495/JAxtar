import os
from pathlib import Path
from typing import Any, Dict

import click
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from cli.evaluation_runner import run_evaluation_sweep
from config.pydantic_models import DistTrainOptions, EvalOptions, PuzzleOptions
from helpers.config_printer import print_config
from helpers.logger import create_logger
from helpers.rich_progress import trange
from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.qstar import qstar_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder
from train_util.optimizer import get_eval_params, get_learning_rate, setup_optimizer
from train_util.target_update import scaled_by_reset, soft_update

from ..options import (
    dist_heuristic_options,
    dist_puzzle_options,
    dist_qfunction_options,
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
    k_max: int,
    eval_options: EvalOptions,
    heuristic_config: Dict[str, Any],
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

    steps = train_options.steps // train_options.replay_ratio
    update_interval = train_options.update_interval // train_options.replay_ratio
    reset_interval = train_options.reset_interval // train_options.replay_ratio
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
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
        td_error_clip=train_options.td_error_clip,
    )
    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_model,
        train_options.dataset_batch_size,
        k_max,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        train_options.using_triangular_sampling,
        n_devices=n_devices,
        temperature=train_options.temperature,
        td_error_clip=train_options.td_error_clip,
        use_diffusion_distance=train_options.use_diffusion_distance,
        use_diffusion_distance_mixture=train_options.use_diffusion_distance_mixture,
        use_diffusion_distance_warmup=train_options.use_diffusion_distance_warmup,
        diffusion_distance_warmup_steps=train_options.diffusion_distance_warmup_steps,
        non_backtracking_steps=train_options.sampling_non_backtracking_steps,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update
    eval_params = get_eval_params(opt_state, heuristic_params)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, eval_params, subkey, i)
        target_heuristic = dataset["target_heuristic"]
        mean_target_heuristic = jnp.mean(target_heuristic)
        mean_target_entropy = None
        if "target_entropy" in dataset:
            mean_target_entropy = jnp.mean(dataset["target_entropy"])

        (
            heuristic_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
            current_heuristics,
        ) = davi_fn(key, dataset, heuristic_params, opt_state)
        eval_params = get_eval_params(opt_state, heuristic_params)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        lr = get_learning_rate(opt_state)
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
        if mean_target_entropy is not None:
            logger.log_scalar("Metrics/Mean Target Entropy", mean_target_entropy, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_heuristic, i)
            logger.log_histogram("Metrics/Current Heuristic", current_heuristics, i)
            if "target_entropy" in dataset:
                logger.log_histogram("Metrics/Target Entropy", dataset["target_entropy"], i)

        target_updated = False
        if train_options.use_soft_update:
            target_heuristic_params = soft_update(
                target_heuristic_params, eval_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            if i % update_interval == 0 and i != 0:
                target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_heuristic_params = eval_params
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
            heuristic.params = eval_params
            backup_path = os.path.join(logger.log_dir, f"heuristic_{i}.pkl")
            heuristic.save_model(path=backup_path)
            # Log model as artifact
            if eval_options.num_eval > 0:
                light_eval_options = eval_options.light_eval_options
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                with pbar.pause():
                    run_evaluation_sweep(
                        puzzle=puzzle,
                        puzzle_name=puzzle_name,
                        search_model=heuristic,
                        search_model_name="heuristic",
                        run_label="astar",
                        search_builder_fn=astar_builder,
                        eval_options=light_eval_options,
                        puzzle_opts=puzzle_opts,
                        output_dir=eval_run_dir,
                        logger=logger,
                        step=i,
                        **kwargs,
                    )
    heuristic.params = eval_params
    backup_path = os.path.join(logger.log_dir, "heuristic_final.pkl")
    heuristic.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "heuristic_final", "model")

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        with pbar.pause():
            run_evaluation_sweep(
                puzzle=puzzle,
                puzzle_name=puzzle_name,
                search_model=heuristic,
                search_model_name="heuristic",
                run_label="astar",
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
    k_max: int,
    eval_options: EvalOptions,
    q_config: Dict[str, Any],
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
        loss_type=train_options.loss,
        loss_args=train_options.loss_args,
        replay_ratio=train_options.replay_ratio,
        td_error_clip=train_options.td_error_clip,
    )
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_model,
        train_options.dataset_batch_size,
        k_max,
        train_options.dataset_minibatch_size,
        train_options.using_hindsight_target,
        train_options.using_triangular_sampling,
        n_devices=n_devices,
        temperature=train_options.temperature,
        td_error_clip=train_options.td_error_clip,
        use_double_dqn=train_options.use_double_dqn,
        use_diffusion_distance=train_options.use_diffusion_distance,
        use_diffusion_distance_mixture=train_options.use_diffusion_distance_mixture,
        use_diffusion_distance_warmup=train_options.use_diffusion_distance_warmup,
        diffusion_distance_warmup_steps=train_options.diffusion_distance_warmup_steps,
        non_backtracking_steps=train_options.sampling_non_backtracking_steps,
    )

    pbar = trange(steps)
    updated = False
    last_reset_time = 0
    last_update_step = -1  # Track last update step for force update
    eval_params = get_eval_params(opt_state, qfunc_params)
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, eval_params, subkey, i)
        target_q = dataset["target_q"]
        mean_target_q = jnp.mean(target_q)
        # Optional: mean action entropy when using policy sampling
        mean_action_entropy = None
        mean_target_entropy = None
        if "action_entropy" in dataset:
            mean_action_entropy = jnp.mean(dataset["action_entropy"])
        if "target_entropy" in dataset:
            mean_target_entropy = jnp.mean(dataset["target_entropy"])

        (
            qfunc_params,
            opt_state,
            loss,
            grad_magnitude,
            weight_magnitude,
            diffs,
            current_qs,
        ) = qlearning_fn(key, dataset, qfunc_params, opt_state)
        eval_params = get_eval_params(opt_state, qfunc_params)
        mean_abs_diff = jnp.mean(jnp.abs(diffs))
        lr = get_learning_rate(opt_state)
        pbar.set_description(
            desc="Q-Learning Training",
            desc_dict={
                "lr": lr,
                "loss": float(loss),
                "abs_diff": float(mean_abs_diff),
                "target_q": float(mean_target_q),
                **(
                    {"entropy": float(mean_action_entropy)}
                    if mean_action_entropy is not None
                    else {}
                ),
            },
        )

        logger.log_scalar("Metrics/Learning Rate", lr, i)
        logger.log_scalar("Losses/Loss", loss, i)
        logger.log_scalar("Losses/Mean Abs Diff", mean_abs_diff, i)
        logger.log_scalar("Metrics/Mean Target", mean_target_q, i)
        logger.log_scalar("Metrics/Magnitude Gradient", grad_magnitude, i)
        logger.log_scalar("Metrics/Magnitude Weight", weight_magnitude, i)
        if mean_action_entropy is not None:
            logger.log_scalar("Metrics/Mean Action Entropy", mean_action_entropy, i)
        if mean_target_entropy is not None:
            logger.log_scalar("Metrics/Mean Target Entropy", mean_target_entropy, i)
        if i % 100 == 0:
            logger.log_histogram("Losses/Diff", diffs, i)
            logger.log_histogram("Metrics/Target", target_q, i)
            if "action_entropy" in dataset:
                logger.log_histogram("Metrics/Action Entropy", dataset["action_entropy"], i)
            if "target_entropy" in dataset:
                logger.log_histogram("Metrics/Target Entropy", dataset["target_entropy"], i)
            logger.log_histogram("Metrics/Current Q", current_qs, i)

        target_updated = False
        if train_options.use_soft_update:
            target_qfunc_params = soft_update(
                target_qfunc_params, eval_params, float(1 - 1.0 / update_interval)
            )
            updated = True
            if i % update_interval == 0 and i != 0:
                target_updated = True
        elif ((i % update_interval == 0 and i != 0) and loss <= train_options.loss_threshold) or (
            i - last_update_step >= train_options.force_update_interval
        ):
            target_qfunc_params = eval_params
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
            qfunction.params = eval_params
            backup_path = os.path.join(logger.log_dir, f"qfunction_{i}.pkl")
            qfunction.save_model(path=backup_path)
            # Log model as artifact
            if eval_options.num_eval > 0:
                light_eval_options = eval_options.light_eval_options
                eval_run_dir = Path(logger.log_dir) / "evaluation" / f"step_{i}"
                with pbar.pause():
                    run_evaluation_sweep(
                        puzzle=puzzle,
                        puzzle_name=puzzle_name,
                        search_model=qfunction,
                        search_model_name="qfunction",
                        run_label="qstar",
                        search_builder_fn=qstar_builder,
                        eval_options=light_eval_options,
                        puzzle_opts=puzzle_opts,
                        output_dir=eval_run_dir,
                        logger=logger,
                        step=i,
                        **kwargs,
                    )
    qfunction.params = eval_params
    backup_path = os.path.join(logger.log_dir, "qfunction_final.pkl")
    qfunction.save_model(path=backup_path)
    # Log final model as artifact
    logger.log_artifact(backup_path, "qfunction_final", "model")

    # Evaluation
    if eval_options.num_eval > 0:
        eval_run_dir = Path(logger.log_dir) / "evaluation"
        with pbar.pause():
            run_evaluation_sweep(
                puzzle=puzzle,
                puzzle_name=puzzle_name,
                search_model=qfunction,
                search_model_name="qfunction",
                run_label="qstar",
                search_builder_fn=qstar_builder,
                eval_options=eval_options,
                puzzle_opts=puzzle_opts,
                output_dir=eval_run_dir,
                logger=logger,
                step=steps,
                **kwargs,
            )

    logger.close()

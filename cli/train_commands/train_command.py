import math
import os
from datetime import datetime
from typing import Any

import click
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorboardX
from tqdm import trange

from heuristic.neuralheuristic.davi import davi_builder, get_heuristic_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.puzzle_base import Puzzle
from puzzle.world_model.world_model_train import get_world_model_dataset_builder
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase
from qfunction.neuralq.qlearning import get_qlearning_dataset_builder, qlearning_builder

from .train_option import (
    dataset_options,
    heuristic_options,
    puzzle_options,
    qfunction_options,
    train_option,
)

PyTree = Any


def setup_logging(puzzle_name: str, puzzle_size: int) -> tensorboardX.SummaryWriter:
    log_dir = f"runs/{puzzle_name}_{puzzle_size}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tensorboardX.SummaryWriter(log_dir)


def setup_optimizer(params: PyTree) -> optax.OptState:
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients to a maximum global norm of 1.0
        optax.adam(1e-3, nesterov=True),
    )
    return optimizer, optimizer.init(params)


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
    key: int,
    loss_threshold: float,
    **kwargs,
):

    writer = setup_logging(puzzle_name, puzzle_size)
    heuristic_fn = heuristic.model.apply
    heuristic_params = heuristic.get_new_params()
    target_heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(heuristic_params)
    davi_fn = davi_builder(1000, heuristic_fn, optimizer)
    get_datasets = get_heuristic_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_fn,
        int(1e5),
        int(math.ceil(10000 / shuffle_length)),
        shuffle_length,
        10000,
    )

    pbar = trange(steps)
    mean_target_heuristic = 0
    save_count = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_heuristic_params, subkey)
        heuristic_params, opt_state, loss, mean_abs_diff, diffs = davi_fn(
            key, dataset, heuristic_params, opt_state
        )
        pbar.set_description(
            f"loss: {loss:.4f}, mean_abs_diff: {mean_abs_diff:.2f}, mean_target_heuristic: {mean_target_heuristic:.4f}"
        )
        if i % 100 == 0:
            target_heuristic = dataset[1]
            mean_target_heuristic = jnp.mean(target_heuristic)
            writer.add_scalar("Mean Target Heuristic", mean_target_heuristic, i)
            writer.add_histogram("Target Heuristic", target_heuristic, i)
            writer.add_histogram("Diff", diffs, i)
            writer.add_scalar("Loss", loss, i)
            writer.add_scalar("Mean Abs Diff", mean_abs_diff, i)

        if (i % 500 == 0 and i != 0) and loss <= loss_threshold:
            save_count += 1
            target_heuristic_params, heuristic_params = (heuristic_params, target_heuristic_params)
            opt_state = optimizer.init(heuristic_params)

            if save_count >= 5:
                heuristic.params = target_heuristic_params
                heuristic.save_model(
                    f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
                )
                save_count = 0


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
    key: int,
    loss_threshold: float,
    **kwargs,
):
    writer = setup_logging(puzzle_name, puzzle_size)
    qfunc_fn = qfunction.model.apply
    qfunc_params = qfunction.get_new_params()
    target_qfunc_params = qfunction.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)

    optimizer, opt_state = setup_optimizer(qfunc_params)
    qlearning_fn = qlearning_builder(1000, qfunc_fn, optimizer)
    get_datasets = get_qlearning_dataset_builder(
        puzzle,
        qfunction.pre_process,
        qfunc_fn,
        int(1e5),
        int(math.ceil(10000 / shuffle_length)),
        shuffle_length,
        10000,
    )

    pbar = trange(steps)
    mean_target_heuristic = 0
    save_count = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(target_qfunc_params, subkey)
        qfunc_params, opt_state, loss, mean_abs_diff, diffs = qlearning_fn(
            key, dataset, qfunc_params, opt_state
        )
        pbar.set_description(
            f"loss: {loss:.4f}, mean_abs_diff: {mean_abs_diff:.2f}, mean_target_heuristic: {mean_target_heuristic:.4f}"
        )
        if i % 100 == 0:
            target_heuristic = dataset[1]
            mean_target_heuristic = jnp.mean(target_heuristic)
            writer.add_scalar("Mean Target Heuristic", mean_target_heuristic, i)
            writer.add_histogram("Target Heuristic", target_heuristic, i)
            writer.add_histogram("Diff", diffs, i)
            writer.add_scalar("Loss", loss, i)
            writer.add_scalar("Mean Abs Diff", mean_abs_diff, i)

        if (i % 500 == 0 and i != 0) and loss <= loss_threshold:
            save_count += 1
            target_qfunc_params, qfunc_params = (qfunc_params, target_qfunc_params)
            opt_state = optimizer.init(qfunc_params)

            if save_count >= 5:
                qfunction.params = target_qfunc_params
                qfunction.save_model(
                    f"qfunction/neuralq/model/params/{puzzle_name}_{puzzle_size}.pkl"
                )
                save_count = 0


@click.command()
@puzzle_options
@dataset_options
def make_puzzle_dataset(
    puzzle: Puzzle,
    puzzle_name: str,
    dataset_size: int,
    dataset_minibatch_size: int,
    shuffle_length: int,
    img_size: tuple,
    key: int,
    **kwargs,
):
    shuffle_parallel = int(math.ceil(dataset_minibatch_size / shuffle_length))
    get_datasets = get_world_model_dataset_builder(
        puzzle, dataset_size, shuffle_parallel, shuffle_length, dataset_minibatch_size
    )

    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    np.save(f"tmp/{puzzle_name}/actions.npy", dataset[1])
    images_stack = []
    next_images_stack = []
    for i in trange(len(dataset[0])):
        state_img = dataset[0][i].img()
        next_state_img = dataset[2][i].img()
        small_state_img = cv2.resize(state_img, img_size, interpolation=cv2.INTER_AREA)
        small_next_state_img = cv2.resize(next_state_img, img_size, interpolation=cv2.INTER_AREA)
        if i < 3:
            cv2.imwrite(
                f"tmp/{puzzle_name}/state_img_{i}.png", cv2.cvtColor(state_img, cv2.COLOR_BGR2RGB)
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/next_state_img_{i}.png",
                cv2.cvtColor(next_state_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/small_state_img_{i}.png",
                cv2.cvtColor(small_state_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/small_next_state_img_{i}.png",
                cv2.cvtColor(small_next_state_img, cv2.COLOR_BGR2RGB),
            )

        images_stack.append(small_state_img)
        next_images_stack.append(small_next_state_img)
    images_stack = np.stack(images_stack, axis=0)
    next_images_stack = np.stack(next_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/images.npy", images_stack)
    np.save(f"tmp/{puzzle_name}/next_images.npy", next_images_stack)

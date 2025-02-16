import math
import os

import click
import cv2
import jax
import numpy as np
from tqdm import trange

from cli.train_commands.train_option import dataset_options, puzzle_ds_options
from puzzle.puzzle_base import Puzzle
from puzzle.world_model.world_model_ds import (
    get_sample_data_builder,
    get_world_model_dataset_builder,
)


@click.command()
@puzzle_ds_options
@dataset_options
def make_puzzle_transition_dataset(
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


@click.command()
@puzzle_ds_options
@dataset_options
def make_puzzle_sample_data(
    puzzle: Puzzle,
    puzzle_name: str,
    dataset_size: int,
    dataset_minibatch_size: int,
    img_size: tuple,
    key: int,
    **kwargs,
):
    shuffle_parallel = int(math.ceil(dataset_minibatch_size))
    get_datasets = get_sample_data_builder(puzzle, dataset_size, shuffle_parallel)

    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    target_images_stack = []
    initial_images_stack = []
    for i in trange(len(dataset[0])):
        target_img = dataset[0][i].img()
        initial_img = dataset[1][i].img()
        small_target_img = cv2.resize(target_img, img_size, interpolation=cv2.INTER_AREA)
        small_initial_img = cv2.resize(initial_img, img_size, interpolation=cv2.INTER_AREA)
        if i < 3:
            cv2.imwrite(
                f"tmp/{puzzle_name}/initial_img_{i}.png",
                cv2.cvtColor(initial_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/target_img_{i}.png",
                cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/small_initial_img_{i}.png",
                cv2.cvtColor(small_initial_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/small_target_img_{i}.png",
                cv2.cvtColor(small_target_img, cv2.COLOR_BGR2RGB),
            )

        initial_images_stack.append(small_initial_img)
        target_images_stack.append(small_target_img)
    initial_images_stack = np.stack(initial_images_stack, axis=0)
    target_images_stack = np.stack(target_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/initial_images.npy", initial_images_stack)
    np.save(f"tmp/{puzzle_name}/target_images.npy", target_images_stack)

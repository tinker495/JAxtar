import math
import os

import click
import cv2
import jax
import numpy as np
from puxle import Puzzle
from tqdm import trange

from cli.train_commands.world_model_train_option import (
    dataset_options,
    puzzle_ds_options,
)
from world_model_puzzle.world_model_ds import (
    create_eval_trajectory,
    get_sample_data_builder,
    get_world_model_dataset_builder,
)


def convert_to_imgs(state: Puzzle.State, img_size: tuple):
    state_img = state.img()
    small_state_img = cv2.resize(state_img, img_size, interpolation=cv2.INTER_AREA)
    return state_img, small_state_img


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
    states, actions, next_states = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    np.save(f"tmp/{puzzle_name}/actions.npy", actions)
    images_stack = []
    next_images_stack = []
    for i in trange(len(actions)):
        state_img, small_state_img = convert_to_imgs(states[i], img_size)
        next_state_img, small_next_state_img = convert_to_imgs(next_states[i], img_size)
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
    target_states, initial_states = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    target_images_stack = []
    initial_images_stack = []
    for i in trange(len(target_states)):
        target_img, small_target_img = convert_to_imgs(target_states[i], img_size)
        initial_img, small_initial_img = convert_to_imgs(initial_states[i], img_size)
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
    np.save(f"tmp/{puzzle_name}/inits.npy", initial_images_stack)
    np.save(f"tmp/{puzzle_name}/targets.npy", target_images_stack)


@click.command()
@puzzle_ds_options
@dataset_options
def make_puzzle_eval_trajectory(
    puzzle: Puzzle,
    puzzle_name: str,
    img_size: tuple,
    key: int,
    **kwargs,
):

    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    shuffle_length = 10000
    states, actions = create_eval_trajectory(puzzle, shuffle_length, key)
    print(f"states.shape: {states.shape}")
    print(f"actions.shape: {actions.shape}")

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    np.save(f"tmp/{puzzle_name}/eval_actions.npy", actions)
    state_images_stack = []
    for i in trange(len(states)):
        state_img, small_state_img = convert_to_imgs(states[i], img_size)
        if i < 3:
            cv2.imwrite(
                f"tmp/{puzzle_name}/state_img_{i}.png",
                cv2.cvtColor(state_img, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                f"tmp/{puzzle_name}/small_state_img_{i}.png",
                cv2.cvtColor(small_state_img, cv2.COLOR_BGR2RGB),
            )
        state_images_stack.append(small_state_img)
    state_images_stack = np.stack(state_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/eval_traj_images.npy", state_images_stack)

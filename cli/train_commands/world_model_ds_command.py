import math
import os

import click
import cv2
import jax
import numpy as np
from puxle import Puzzle

from config.pydantic_models import WMDatasetOptions
from helpers.config_printer import print_config
from helpers.logger import create_logger
from helpers.rich_progress import trange
from world_model_puzzle.world_model_ds import (
    create_eval_trajectory,
    get_sample_data_builder,
    get_world_model_dataset_builder,
)

from ..options import wm_dataset_options, wm_puzzle_ds_options


def convert_to_imgs(state: Puzzle.State, img_size: tuple):
    state_img = state.img()
    small_state_img = cv2.resize(state_img, img_size, interpolation=cv2.INTER_AREA)
    return state_img, small_state_img


@click.command()
@wm_puzzle_ds_options
@wm_dataset_options
def make_puzzle_transition_dataset(
    puzzle: Puzzle,
    puzzle_name: str,
    wm_dataset_options: WMDatasetOptions,
    **kwargs,
):
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "puzzle_name": puzzle_name,
        "wm_dataset_options": wm_dataset_options,
        **kwargs,
    }
    print_config("Make Puzzle Transition Dataset Configuration", config)
    logger = create_logger("aim", f"{puzzle_name}_make_transition_dataset", config)

    shuffle_parallel = int(
        math.ceil(wm_dataset_options.dataset_minibatch_size / wm_dataset_options.shuffle_length)
    )
    get_datasets = get_world_model_dataset_builder(
        puzzle,
        wm_dataset_options.dataset_size,
        shuffle_parallel,
        wm_dataset_options.shuffle_length,
        wm_dataset_options.dataset_minibatch_size,
    )

    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if wm_dataset_options.key == 0 else wm_dataset_options.key
    )
    key, subkey = jax.random.split(key)
    states, actions, next_states = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    np.save(f"tmp/{puzzle_name}/actions.npy", actions)
    images_stack = []
    next_images_stack = []
    for i in trange(len(actions)):
        state_img, small_state_img = convert_to_imgs(states[i], wm_dataset_options.img_size)
        next_state_img, small_next_state_img = convert_to_imgs(
            next_states[i], wm_dataset_options.img_size
        )
        if i < 3:
            logger.log_image(f"State/{i}", cv2.cvtColor(small_state_img, cv2.COLOR_BGR2RGB), i)
            logger.log_image(
                f"Next_State/{i}", cv2.cvtColor(small_next_state_img, cv2.COLOR_BGR2RGB), i
            )

        images_stack.append(small_state_img)
        next_images_stack.append(small_next_state_img)
    images_stack = np.stack(images_stack, axis=0)
    next_images_stack = np.stack(next_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/images.npy", images_stack)
    np.save(f"tmp/{puzzle_name}/next_images.npy", next_images_stack)
    logger.close()


@click.command()
@wm_puzzle_ds_options
@wm_dataset_options
def make_puzzle_sample_data(
    puzzle: Puzzle,
    puzzle_name: str,
    wm_dataset_options: WMDatasetOptions,
    **kwargs,
):
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "puzzle_name": puzzle_name,
        "wm_dataset_options": wm_dataset_options,
        **kwargs,
    }
    print_config("Make Puzzle Sample Data Configuration", config)
    logger = create_logger("aim", f"{puzzle_name}_make_sample_data", config)
    shuffle_parallel = int(math.ceil(wm_dataset_options.dataset_minibatch_size))
    get_datasets = get_sample_data_builder(
        puzzle, wm_dataset_options.dataset_size, shuffle_parallel
    )

    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if wm_dataset_options.key == 0 else wm_dataset_options.key
    )
    key, subkey = jax.random.split(key)
    target_states, initial_states = get_datasets(subkey)

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    target_images_stack = []
    initial_images_stack = []
    for i in trange(len(target_states)):
        target_img, small_target_img = convert_to_imgs(
            target_states[i], wm_dataset_options.img_size
        )
        initial_img, small_initial_img = convert_to_imgs(
            initial_states[i], wm_dataset_options.img_size
        )
        if i < 3:
            logger.log_image(
                f"Initial_State/{i}", cv2.cvtColor(small_initial_img, cv2.COLOR_BGR2RGB), i
            )
            logger.log_image(
                f"Target_State/{i}", cv2.cvtColor(small_target_img, cv2.COLOR_BGR2RGB), i
            )

        initial_images_stack.append(small_initial_img)
        target_images_stack.append(small_target_img)
    initial_images_stack = np.stack(initial_images_stack, axis=0)
    target_images_stack = np.stack(target_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/inits.npy", initial_images_stack)
    np.save(f"tmp/{puzzle_name}/targets.npy", target_images_stack)
    logger.close()


@click.command()
@wm_puzzle_ds_options
@wm_dataset_options
def make_puzzle_eval_trajectory(
    puzzle: Puzzle,
    puzzle_name: str,
    wm_dataset_options: WMDatasetOptions,
    **kwargs,
):
    config = {
        "puzzle": {"name": puzzle_name, "size": puzzle.size},
        "puzzle_name": puzzle_name,
        "wm_dataset_options": wm_dataset_options,
        **kwargs,
    }
    print_config("Make Puzzle Eval Trajectory Configuration", config)
    logger = create_logger("aim", f"{puzzle_name}_make_eval_trajectory", config)

    key = jax.random.PRNGKey(
        np.random.randint(0, 1000000) if wm_dataset_options.key == 0 else wm_dataset_options.key
    )
    shuffle_length = 10000
    states, actions = create_eval_trajectory(puzzle, shuffle_length, key)
    print(f"states.shape: {states.shape}")
    print(f"actions.shape: {actions.shape}")

    os.makedirs(f"tmp/{puzzle_name}", exist_ok=True)
    np.save(f"tmp/{puzzle_name}/eval_actions.npy", actions)
    state_images_stack = []
    for i in trange(len(states)):
        state_img, small_state_img = convert_to_imgs(states[i], wm_dataset_options.img_size)
        if i < 3:
            logger.log_image(f"State/{i}", cv2.cvtColor(small_state_img, cv2.COLOR_BGR2RGB), i)
        state_images_stack.append(small_state_img)
    state_images_stack = np.stack(state_images_stack, axis=0)
    np.save(f"tmp/{puzzle_name}/eval_traj_images.npy", state_images_stack)
    logger.close()

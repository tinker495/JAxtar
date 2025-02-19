from functools import wraps

import click
import jax.numpy as jnp

from config import (
    default_puzzle_sizes,
    puzzle_dict_ds,
    world_model_dict,
    world_model_ds_dict,
)


def get_ds_options(func: callable) -> callable:
    @click.option(
        "-ds",
        "--dataset",
        default="rubikscube",
        type=click.Choice(world_model_ds_dict.keys()),
        help="Dataset to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        dataset_name = kwargs["dataset"]
        dataset_path = world_model_ds_dict[dataset_name]

        datas = jnp.load(dataset_path + "/images.npy")
        next_datas = jnp.load(dataset_path + "/next_images.npy")
        actions = jnp.load(dataset_path + "/actions.npy")
        kwargs["datas"] = datas
        kwargs["next_datas"] = next_datas
        kwargs["actions"] = actions

        eval_trajectory = jnp.load(dataset_path + "/eval_traj_images.npy")
        eval_actions = jnp.load(dataset_path + "/eval_actions.npy")
        kwargs["eval_trajectory"] = (eval_trajectory, eval_actions)
        return func(*args, **kwargs)

    return wrapper


def get_world_model_options(func: callable) -> callable:
    @click.option(
        "--world_model",
        default="rubikscube",
        type=click.Choice(world_model_dict.keys()),
        help="World model to use",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        world_model_name = kwargs["world_model"]
        world_model = world_model_dict[world_model_name]
        kwargs["world_model"] = world_model(reset=True)
        return func(*args, **kwargs)

    return wrapper


def train_options(func: callable) -> callable:
    @click.option("--train_epochs", type=int, default=2000, help="Number of training steps")
    @click.option("--mini_batch_size", type=int, default=1000, help="Batch size")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def puzzle_ds_options(func: callable) -> callable:
    @click.option(
        "-p",
        "--puzzle",
        default="rubikscube",
        type=click.Choice(puzzle_dict_ds.keys()),
        help="Puzzle to solve",
    )
    @click.option("-ps", "--puzzle_size", default="default", type=str, help="Size of the puzzle")
    @wraps(func)
    def wrapper(*args, **kwargs):
        puzzle_name = kwargs["puzzle"]
        puzzle_size = kwargs["puzzle_size"]
        if puzzle_size == "default":
            puzzle_size = default_puzzle_sizes[puzzle_name]
        else:
            puzzle_size = int(puzzle_size)

        kwargs["puzzle"] = puzzle_dict_ds[puzzle_name](size=puzzle_size)
        kwargs["puzzle_name"] = puzzle_name
        kwargs["puzzle_size"] = puzzle_size
        return func(*args, **kwargs)

    return wrapper


def dataset_options(func: callable) -> callable:
    @click.option("--dataset_size", type=int, default=300000)
    @click.option("--dataset_minibatch_size", type=int, default=30000)
    @click.option("--shuffle_length", type=int, default=30)
    @click.option("--img_size", nargs=2, type=click.Tuple([int, int]), default=(32, 32))
    @click.option("--key", type=int, default=0)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

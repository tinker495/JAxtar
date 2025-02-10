import math
from typing import Any

import click
import jax
import numpy as np

from puzzle.neuralpuzzle.get_wold_model_ds import get_dataset_builder
from puzzle_config import default_puzzle_sizes, puzzle_dict

PyTree = Any


@click.command()
@click.option(
    "-p",
    "--puzzle",
    default="n-puzzle",
    type=click.Choice(puzzle_dict.keys()),
    help="Puzzle to solve",
)
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option(
    "--shuffle_length", type=int, default=30, help="Shuffle length for dataset generation"
)
@click.option("--key", type=int, default=0, help="Random seed key (0 for random)")
@click.option("--debug", is_flag=True, help="Debug mode (disables JIT)")
@click.option(
    "--output", type=str, default="world_model_dataset.npz", help="Output filename for the dataset"
)
def make_world_model_ds(
    puzzle: str,
    puzzle_size: str,
    shuffle_length: int,
    key: int,
    debug: bool,
    output: str,
):
    """
    Generate a dataset for the world model and save it to a file.

    The dataset consists of tiled targets, states, actions, and next_states,
    generated using the provided puzzle and neural function.
    """
    if debug:
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle_size)

    puzzle_name = puzzle
    puzzle_instance = puzzle_dict[puzzle_name](puzzle_size)

    # Parameters for dataset generation
    dataset_size = int(1e5)
    dataset_minibatch_size = 10000
    shuffle_parallel = int(math.ceil(dataset_minibatch_size / shuffle_length))

    get_datasets = get_dataset_builder(
        puzzle_instance,
        dataset_size,
        shuffle_parallel,
        shuffle_length,
        dataset_minibatch_size,
    )

    # Initialize random key and generate the dataset
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset = get_datasets(subkey)

    # Save the dataset to a compressed NumPy file
    np.savez_compressed(
        output,
        states=dataset[0],
        actions=dataset[1],
        next_states=dataset[2],
    )
    print(f"World model dataset saved to {output}")


if __name__ == "__main__":
    make_world_model_ds()

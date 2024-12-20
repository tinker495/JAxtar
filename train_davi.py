import math
from datetime import datetime
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorboardX
from tqdm import trange

from heuristic.neuralheuristic.davi import davi_builder, get_dataset_builder
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle_config import default_puzzle_sizes, puzzle_dict, puzzle_heuristic_dict_nn

PyTree = Any


@jax.jit
def soft_update(new_tensors: PyTree, old_tensors: PyTree, tau: float):
    return jax.tree_util.tree_map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, old_tensors
    )


@click.command()
@click.option(
    "-p",
    "--puzzle",
    default="n-puzzle",
    type=click.Choice(puzzle_heuristic_dict_nn.keys()),
    help="Puzzle to solve",
)
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--steps", type=int, default=1000000)
@click.option("--shuffle_length", type=int, default=30)
@click.option("--key", type=int, default=0)
@click.option("--reset", is_flag=True, help="Reset the target heuristic params")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("-l", "--loss_threshold", type=float, default=0.05)
def train_davi(
    puzzle: str,
    puzzle_size: int,
    steps: int,
    shuffle_length: int,
    key: int,
    reset: bool,
    debug: bool,
    loss_threshold: float,
):
    if debug:
        # disable jit
        print("Disabling JIT")
        jax.config.update("jax_disable_jit", True)
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle_size)
    puzzle_name = puzzle
    puzzle = puzzle_dict[puzzle_name](puzzle_size)
    heuristic: NeuralHeuristicBase = puzzle_heuristic_dict_nn[puzzle_name](
        puzzle_size, puzzle, reset
    )

    # Setup tensorboard logging
    log_dir = f"runs/{puzzle_name}_{puzzle_size}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tensorboardX.SummaryWriter(log_dir)

    heuristic_fn = heuristic.model.apply
    heuristic_params = heuristic.get_new_params()
    target_heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset_size = int(1e5)
    dataset_minibatch_size = 10000
    shuffle_parallel = int(math.ceil(dataset_minibatch_size / shuffle_length))
    minibatch_size = 1000

    optimizer = optax.adabelief(1e-3)
    opt_state = optimizer.init(heuristic_params)

    davi_fn = davi_builder(minibatch_size, heuristic_fn, optimizer)
    get_datasets = get_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_fn,
        dataset_size,
        shuffle_parallel,
        shuffle_length,
        dataset_minibatch_size,
    )

    pbar = trange(steps)
    mean_target_heuristic = 0
    save_count = 0
    for i in pbar:
        key, subkey = jax.random.split(key)
        dataset = get_datasets(
            target_heuristic_params,
            subkey,
        )
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

            # Log metrics to tensorboard
            writer.add_scalar("Loss", loss, i)
            writer.add_scalar("Mean Abs Diff", mean_abs_diff, i)

        if (i % 500 == 0 and i != 0) and loss <= loss_threshold:
            save_count += 1
            # swap target and current params
            target_heuristic_params, heuristic_params = (heuristic_params, target_heuristic_params)
            opt_state = optimizer.init(heuristic_params)  # reset optimizer state

            if save_count >= 5:
                heuristic.params = target_heuristic_params
                heuristic.save_model(
                    f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
                )
                save_count = 0


if __name__ == "__main__":
    train_davi()

from datetime import datetime
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
import tensorboardX
from tqdm import trange

from heuristic.neuralheuristic.davi import davi_builder, get_dataset_builder
from puzzle_config import default_puzzle_sizes, puzzle_dict, puzzle_heuristic_dict_nn

PyTree = Any


def random_split_like_tree(rng_key: jax.random.PRNGKey, target: PyTree = None, treedef=None):
    if treedef is None:
        treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def tree_random_normal_like(rng_key: jax.random.PRNGKey, target: PyTree):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree.map(
        lambda t, k: jax.random.normal(k, t.shape, t.dtype) * jnp.std(t),
        target,
        keys_tree,
    )


def soft_reset(tensors, tau, key):
    new_tensors = tree_random_normal_like(key, tensors)
    soft_reseted = jax.tree.map(
        lambda new, old: tau * new + (1.0 - tau) * old, new_tensors, tensors
    )
    # name dense is hardreset
    return soft_reseted


@click.command()
@click.option(
    "--puzzle",
    default="n-puzzle",
    type=click.Choice(puzzle_heuristic_dict_nn.keys()),
    help="Puzzle to solve",
)
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--steps", type=int, default=100000)
@click.option("--key", type=int, default=0)
@click.option("--reset", is_flag=True, help="Reset the target heuristic params")
@click.option("--debug", is_flag=True, help="Debug mode")
def train_davi(puzzle: str, puzzle_size: int, steps: int, key: int, reset: bool, debug: bool):
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
    heuristic = puzzle_heuristic_dict_nn[puzzle_name](puzzle_size, puzzle, reset)
    heuristic.save_model(f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl")

    # Setup tensorboard logging
    log_dir = f"runs/{puzzle_name}_{puzzle_size}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tensorboardX.SummaryWriter(log_dir)

    heuristic_fn = heuristic.model.apply
    heuristic_params = heuristic.params
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    dataset_size = int(1e7)
    batch_size = int(1e1)
    shuffle_length = 200
    minibatch_size = 128

    davi_fn, opt_state = davi_builder(minibatch_size, heuristic_fn, heuristic_params)
    get_datasets = get_dataset_builder(
        puzzle,
        heuristic.pre_process,
        heuristic_fn,
        dataset_size,
        batch_size,
        shuffle_length,
    )

    pbar = trange(steps)
    dataset = get_datasets(
        heuristic_params,
        subkey,
    )

    heuristic_params = soft_reset(heuristic_params, 0.2, subkey)
    target_heuristic = dataset[1]
    mean_target_heuristic = jnp.mean(target_heuristic)
    writer.add_scalar("Mean Target Heuristic", mean_target_heuristic, 0)
    writer.add_histogram("Target Heuristic", target_heuristic, 0)
    pbar.set_description(f"loss: {0:.4f}, mean_target_heuristic: {mean_target_heuristic:.4f}")
    for i in pbar:
        key, subkey = jax.random.split(key)
        heuristic_params, opt_state, loss = davi_fn(key, dataset, heuristic_params, opt_state)
        pbar.set_description(
            f"loss: {loss:.4f}, mean_target_heuristic: {mean_target_heuristic:.4f}"
        )

        # Log metrics to tensorboard
        writer.add_scalar("Loss", loss, i)

        if i % 2 == 0 and i != 0 and loss < 1e-3:
            heuristic.params = heuristic_params
            heuristic.save_model(
                f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl"
            )
            dataset = get_datasets(
                heuristic_params,
                subkey,
            )
            target_heuristic = dataset[1]
            mean_target_heuristic = jnp.mean(target_heuristic)
            writer.add_scalar("Mean Target Heuristic", mean_target_heuristic, i)
            writer.add_histogram("Target Heuristic", target_heuristic, i)


if __name__ == "__main__":
    train_davi()

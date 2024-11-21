import copy
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from puzzle_config import puzzle_dict, puzzle_heuristic_dict_nn, default_puzzle_sizes
from heuristic.neuralheuristic.davi import davi_builder
from puzzle_config import default_puzzle_sizes, puzzle_dict_nn

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
@click.option("--puzzle", default="n-puzzle", type=click.Choice(puzzle_heuristic_dict_nn.keys()), help="Puzzle to solve")
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

    heuristic_fn = heuristic.param_distance
    heuristic_params = heuristic.params
    target_heuristic_params = copy.deepcopy(heuristic_params)
    key = jax.random.PRNGKey(np.random.randint(0, 1000000) if key == 0 else key)
    key, subkey = jax.random.split(key)
    heuristic_params = soft_reset(heuristic_params, 0.2, subkey)

    davi_fn, opt_state = davi_builder(
        puzzle, int(1e2), int(1e4), 1000, 10000, heuristic_fn, heuristic_params
    )

    count = 0
    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        heuristic_params, opt_state, loss, mean_target_heuristic = davi_fn(
            subkey, target_heuristic_params, heuristic_params, opt_state
        )
        pbar.set_description(
            f"Loss: {loss:5.4f}, Mean Target Heuristic: {mean_target_heuristic:4.1f}"
        )

        if (i % 100 == 0 and i > 0):
            count += 1
            if loss < 1e-1 or count >= 10:
                heuristic.params = heuristic_params
                heuristic.save_model(f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl")
                target_heuristic_params = copy.deepcopy(heuristic_params)
                count = 0
                print("updated target heuristic params")
            heuristic_params = soft_reset(heuristic_params, 0.2, subkey)
    heuristic.params = heuristic_params
    heuristic.save_model(f"heuristic/neuralheuristic/model/params/{puzzle_name}_{puzzle_size}.pkl")


if __name__ == "__main__":
    train_davi()

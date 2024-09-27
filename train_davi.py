import os
import click
import jax
import jax.numpy as jnp
import chex
import copy
from tqdm import trange

from puzzle_config import puzzle_dict_nn, default_puzzle_sizes
from heuristic.DAVI.davi import davi_builder

@click.command()
@click.option("--puzzle", default="n-puzzle", type=click.Choice(puzzle_dict_nn.keys()), help="Puzzle to solve")
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--steps", type=int, default=10000)
@click.option("--key", type=int, default=0)
@click.option("--debug", is_flag=True, help="Debug mode")
def train_davi(puzzle: str, puzzle_size: int, steps: int, key: int, debug: bool):
    if debug:
        #disable jit
        print("Disabling JIT")
        jax.config.update('jax_disable_jit', True)
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle_size)
    puzzle_name = puzzle
    puzzle, heuristic = puzzle_dict_nn[puzzle_name](puzzle_size)
    heuristic.save_model(f"heuristic/DAVI/neuralheuristic/params/{puzzle_name}_{puzzle_size}.pkl")

    heuristic_fn = heuristic.param_distance
    heuristic_params = heuristic.params
    target_heuristic_params = copy.deepcopy(heuristic_params)

    davi_fn, opt_state = davi_builder(puzzle, int(1e2), int(1e4), 1000, 10000, heuristic_fn, heuristic_params)
    key = jax.random.PRNGKey(key)

    pbar = trange(steps)
    for i in pbar:
        key, subkey = jax.random.split(key)
        heuristic_params, opt_state, loss, mean_target_heuristic = davi_fn(subkey, target_heuristic_params, heuristic_params, opt_state)
        pbar.set_description(f"Loss: {loss:5.4f}, Mean Target Heuristic: {mean_target_heuristic:4.1f}")

        if loss < 1e-2 and i % 100 == 0:
            heuristic.params = heuristic_params
            heuristic.save_model(f"heuristic/DAVI/neuralheuristic/params/{puzzle_name}_{puzzle_size}.pkl")
            target_heuristic_params = copy.deepcopy(heuristic_params)
            print("updated target heuristic params")
    heuristic.params = heuristic_params
    heuristic.save_model(f"heuristic/DAVI/neuralheuristic/params/{puzzle_name}_{puzzle_size}.pkl")

if __name__ == "__main__":
    train_davi()
import click
import jax
import jax.numpy as jnp
import chex

from puzzle.slidepuzzle import SlidePuzzle
from puzzle_config import puzzle_dict_nn, default_puzzle_sizes
from heuristic.DAVI.davi import create_shuffled_path

@click.command()
@click.option("--puzzle", default="n-puzzle", type=click.Choice(puzzle_dict_nn.keys()), help="Puzzle to solve")
@click.option("--puzzle_size", default="default", type=str, help="Size of the puzzle")
@click.option("--steps", type=int, default=10)
@click.option("--debug", is_flag=True, help="Debug mode")
def train_davi(puzzle, puzzle_size: int, steps: int, debug: bool):
    if debug:
        #disable jit
        print("Disabling JIT")
        jax.config.update('jax_disable_jit', True)
    if puzzle_size == "default":
        puzzle_size = default_puzzle_sizes[puzzle]
    else:
        puzzle_size = int(puzzle_size)
    puzzle, heuristic = puzzle_dict_nn[puzzle](puzzle_size)

    heuristic_fn = heuristic.param_distance
    heuristic_params = heuristic.params

    print(heuristic_fn(heuristic_params, puzzle.get_target_state(), puzzle.get_target_state()))

    print(create_shuffled_path(puzzle, 200, 50000, jax.random.PRNGKey(0)))

if __name__ == "__main__":
    train_davi()
import jax
import jax.numpy as jnp
import chex

from puzzle.puzzle_base import Puzzle
from puzzle.slidepuzzle import SlidePuzzle
from heuristic.DAVI.davi import create_shuffled_path

def train_davi(puzzle: Puzzle, steps: int):
    """
    Train DAVI on the sliding puzzle problem.
    """
    print(create_shuffled_path(puzzle, 1000, 1000, jax.random.PRNGKey(0)))

if __name__ == "__main__":
    puzzle = SlidePuzzle(4)
    train_davi(puzzle, 10)
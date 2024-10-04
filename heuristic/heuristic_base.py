import chex
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from puzzle.puzzle_base import Puzzle

class Heuristic(ABC):
    puzzle: Puzzle # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    @abstractmethod
    def distance(self, current: Puzzle.State, target: Puzzle.State) -> float:
        pass

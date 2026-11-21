from abc import ABC, abstractmethod
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle


class Heuristic(ABC):
    puzzle: Puzzle
    is_fixed: bool = (
        False  # True if this heuristic is only valid for a fixed target (i.e. it does not
    )
    # support arbitrary solve_config targets / retargeting).

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def prepare_heuristic_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> Union[Puzzle.SolveConfig, Any]:
        """Called once per search before expansion to precompute search-invariant parameters (e.g., neural weights or goal embeddings). Defaults to returning solve_config unchanged."""
        return solve_config

    def batched_distance(
        self, heuristic_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State
    ) -> chex.Array:
        """Return estimated distances to the target with shape (batch_size,)."""
        return jnp.asarray(
            jax.vmap(self.distance, in_axes=(None, 0))(heuristic_parameters, current)
        )

    @abstractmethod
    def distance(
        self, heuristic_parameters: Union[Puzzle.SolveConfig, Any], current: Puzzle.State, /
    ) -> float | jax.Array:
        """Return scalar estimated distance from a single state to the target."""
        pass

import chex
import jax
import jax.numpy as jnp
from puxle import PancakeSorting

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.dtypes import DTYPE


class PancakeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: PancakeSorting, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: PancakeSorting.SolveConfig, current: PancakeSorting.State
    ) -> chex.Array:
        current_flatten_face = current.stack
        # Create a one-hot encoding of the flattened face
        current_one_hot = jax.nn.one_hot(
            current_flatten_face, num_classes=self.puzzle.size
        ).flatten()  # 6 colors in Rubik's Cube
        if self.is_fixed:
            one_hots = current_one_hot
        else:
            target_flatten_face = solve_config.TargetState.stack
            target_one_hot = jax.nn.one_hot(
                target_flatten_face, num_classes=self.puzzle.size
            ).flatten()
            one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

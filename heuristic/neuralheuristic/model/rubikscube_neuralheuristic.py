import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE
from puzzle.rubikscube import RubiksCube


class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = self.puzzle.unpack_faces(current.faces).flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        current_one_hot = jax.nn.one_hot(
            current_flatten_face, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube
        if self.is_fixed:
            one_hots = current_one_hot
        else:
            target_flatten_face = self.puzzle.unpack_faces(solve_config.TargetState.faces).flatten()
            target_one_hot = jax.nn.one_hot(target_flatten_face, num_classes=6).flatten()
            one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

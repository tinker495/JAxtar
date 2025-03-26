import chex
import jax
import jax.numpy as jnp

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from puzzle.rubikscube import RubiksCube


class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, init_params: bool = True):
        super().__init__(puzzle, init_params=init_params)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        target_face = solve_config.TargetState.faces
        flatten_target_face = self.puzzle.unpack_faces(target_face).flatten()  # (3,3,6) -> (54,)
        flatten_face = self.puzzle.unpack_faces(current.faces).flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        concat_face = jnp.concatenate([flatten_face, flatten_target_face])
        one_hot = jax.nn.one_hot(concat_face, num_classes=6).flatten()  # 6 colors in Rubik's Cube
        return (one_hot - 0.5) * 2.0  # normalize to [-1, 1]

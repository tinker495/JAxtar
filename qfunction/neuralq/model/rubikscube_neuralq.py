import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from neural_util.modules import DTYPE
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class RubiksCubeNeuralQ(NeuralQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, neighbor_encoding: bool = False, **kwargs):
        self.neighbor_encoding = neighbor_encoding
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        current_one_hot = jax.nn.one_hot(
            current_flatten_face, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube

        if self.neighbor_encoding:
            neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
            neighbors_flatten_face = jax.vmap(lambda x: x.unpacked.faces.flatten())(
                neighbors
            )  # (6, 3, 3, 6) -> (6, 54)
            # Create a one-hot encoding of the flattened face
            neighbors_one_hot = jax.nn.one_hot(neighbors_flatten_face, num_classes=6).flatten()
            current_one_hot = jnp.concatenate([current_one_hot, neighbors_one_hot], axis=-1)

        if self.is_fixed:
            one_hots = current_one_hot
        else:
            target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
            target_one_hot = jax.nn.one_hot(target_flatten_face, num_classes=6).flatten()
            one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

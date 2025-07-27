import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from neural_util.modules import DTYPE
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class RubiksCubeNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        current_one_hot = jax.nn.one_hot(
            current_flatten_face, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube
        return ((current_one_hot - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


class RubiksCubeRandomNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        current_one_hot = jax.nn.one_hot(
            current_flatten_face, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube
        target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
        target_one_hot = jax.nn.one_hot(target_flatten_face, num_classes=6).flatten()
        one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

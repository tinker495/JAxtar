import chex
import jax
from puxle import RubiksCube

from neural_util.modules import DTYPE
from qfunction.zeroshotq.zeroshotq_base import ZeroshotQFunctionBase


class RubiksCubeZeroshotQ(ZeroshotQFunctionBase):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process_state(self, state: RubiksCube.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        flatten_face = state.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        # Create a one-hot encoding of the flattened face
        one_hot = jax.nn.one_hot(flatten_face, num_classes=6).flatten()  # 6 colors in Rubik's Cube
        return ((one_hot - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

import chex
import jax
from puxle import RubiksCube

from neural_util.basemodel import HLGResMLPModel
from neural_util.preprocessing import (
    rubikscube_pre_process,
    rubikscube_random_pre_process,
)
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class RubiksCubeNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)
        tile_count = puzzle.size * puzzle.size
        self._num_tile_classes = 6 if self._use_color_embedding else 6 * tile_count
        self.metric = puzzle.metric
        super().__init__(puzzle, **kwargs)

    def _one_hot_faces(self, faces: chex.Array) -> chex.Array:
        return jax.nn.one_hot(faces, num_classes=self._num_tile_classes)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.faces_unpacked.flatten()  # (3,3,6) -> (54,)
        return rubikscube_pre_process(
            self._one_hot_faces, self.metric, self.puzzle.size, current_flatten_face
        )


class RubiksCubeRandomNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)
        tile_count = puzzle.size * puzzle.size
        self._num_tile_classes = 6 if self._use_color_embedding else 6 * tile_count
        self.metric = puzzle.metric
        super().__init__(puzzle, **kwargs)

    def _one_hot_faces(self, faces: chex.Array) -> chex.Array:
        return jax.nn.one_hot(faces, num_classes=self._num_tile_classes)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.faces_unpacked.flatten()  # (3,3,6) -> (54,)
        target_flatten_face = solve_config.TargetState.faces_unpacked.flatten()
        return rubikscube_random_pre_process(
            self._one_hot_faces,
            self.metric,
            self.puzzle.size,
            current_flatten_face,
            target_flatten_face,
        )


class RubiksCubeHLGNeuralQ(RubiksCubeNeuralQ):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)


class RubiksCubeRandomHLGNeuralQ(RubiksCubeRandomNeuralQ):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, model=HLGResMLPModel, **kwargs)

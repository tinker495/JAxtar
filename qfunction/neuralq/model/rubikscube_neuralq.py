import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from neural_util.modules import DTYPE
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


def _remove_face_centers(flatten_face: chex.Array, n: int) -> chex.Array:
    """Drop centre stickers from the flattened cube faces."""
    face_area = n * n
    total_len = flatten_face.shape[0]
    if face_area == 0:
        raise ValueError("Cube dimension must be positive.")
    if total_len % face_area != 0:
        raise ValueError("Flattened face length is incompatible with cube size.")

    if n % 2 == 0:
        return flatten_face

    num_faces = total_len // face_area
    # Compute the (row-major) position of the face centre for the given n.
    centre_index = (n // 2) * n + (n // 2)
    indices_before = jnp.arange(centre_index, dtype=jnp.int32)
    indices_after = jnp.arange(centre_index + 1, face_area, dtype=jnp.int32)
    gather_indices = jnp.concatenate([indices_before, indices_after], axis=0)

    faces = flatten_face.reshape((num_faces, face_area))
    faces_without_centre = jnp.take(faces, gather_indices, axis=1)
    return faces_without_centre.reshape(num_faces * (face_area - 1))


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
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        if self.metric == "UQTM":
            # UQTM need to use all the stickers
            current_one_hot = self._one_hot_faces(current_flatten_face).flatten()
        else:
            current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
            # Create a one-hot encoding of the flattened face without centre stickers
            current_one_hot = self._one_hot_faces(current_no_centers).flatten()
        return ((current_one_hot - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


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
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        if self.metric == "UQTM":
            # UQTM need to use all the stickers
            current_one_hot = self._one_hot_faces(current_flatten_face).flatten()
        else:
            current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
            # Create a one-hot encoding of the flattened face without centre stickers
            current_one_hot = self._one_hot_faces(current_no_centers).flatten()
        target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
        if self.metric == "UQTM":
            # UQTM need to use all the stickers
            target_one_hot = self._one_hot_faces(target_flatten_face).flatten()
        else:
            target_no_centers = _remove_face_centers(target_flatten_face, self.puzzle.size)
            # Create a one-hot encoding of the flattened face without centre stickers
            target_one_hot = self._one_hot_faces(target_no_centers).flatten()
        one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

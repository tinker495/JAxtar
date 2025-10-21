import chex
import jax.numpy as jnp
from puxle import RubiksCube

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
        kwargs.setdefault("num_classes", self._num_tile_classes)
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
        current_tokens = current_no_centers.astype(jnp.int32)
        return current_tokens


class RubiksCubeRandomNeuralQ(NeuralQFunctionBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)
        tile_count = puzzle.size * puzzle.size
        self._num_tile_classes = 6 if self._use_color_embedding else 6 * tile_count
        kwargs.setdefault("num_classes", self._num_tile_classes)
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
        current_tokens = current_no_centers.astype(jnp.int32)
        target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
        target_no_centers = _remove_face_centers(target_flatten_face, self.puzzle.size)
        target_tokens = target_no_centers.astype(jnp.int32)
        return jnp.concatenate([target_tokens, current_tokens], axis=-1)

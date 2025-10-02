import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE


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
    centre_index = (n // 2) * n + (n // 2)
    indices_before = jnp.arange(centre_index, dtype=jnp.int32)
    indices_after = jnp.arange(centre_index + 1, face_area, dtype=jnp.int32)
    gather_indices = jnp.concatenate([indices_before, indices_after], axis=0)

    faces = flatten_face.reshape((num_faces, face_area))
    faces_without_centre = jnp.take(faces, gather_indices, axis=1)
    return faces_without_centre.reshape(num_faces * (face_area - 1))


class RubiksCubeNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
        # Create a one-hot encoding of the flattened face without centre stickers
        current_one_hot = jax.nn.one_hot(
            current_no_centers, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube
        return ((current_one_hot - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


class RubiksCubeRandomNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()  # (3,3,6) -> (54,)
        current_no_centers = _remove_face_centers(current_flatten_face, self.puzzle.size)
        # Create a one-hot encoding of the flattened face without centre stickers
        current_one_hot = jax.nn.one_hot(
            current_no_centers, num_classes=6
        ).flatten()  # 6 colors in Rubik's Cube
        target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
        target_no_centers = _remove_face_centers(target_flatten_face, self.puzzle.size)
        target_one_hot = jax.nn.one_hot(target_no_centers, num_classes=6).flatten()
        one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]

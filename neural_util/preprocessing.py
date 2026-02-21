"""
Shared preprocessing functions for neural heuristic and Q-function models.

These functions are used identically by both heuristic and qfunction implementations,
so they are centralized here to avoid code duplication.
"""

import chex
import jax
import jax.numpy as jnp

from neural_util.dtypes import DTYPE


def lightsout_pre_process(
    current_board_unpacked: chex.Array,
    target_board_unpacked: chex.Array | None,
    is_fixed: bool,
) -> chex.Array:
    """Pre-process LightsOut state for neural models.

    Args:
        current_board_unpacked: The unpacked board of the current state.
        target_board_unpacked: The unpacked board of the target state (None if fixed target).
        is_fixed: Whether the puzzle has a fixed target.

    Returns:
        Normalized array in [-1, 1].
    """
    current_map = current_board_unpacked.astype(DTYPE)
    if is_fixed:
        one_hots = current_map
    else:
        target_map = target_board_unpacked.astype(DTYPE)
        one_hots = jnp.concatenate([target_map, current_map], axis=-1)
    return ((one_hots - 0.5) * 2.0).astype(DTYPE)


def pancake_pre_process(
    current_stack: chex.Array,
    target_stack: chex.Array | None,
    puzzle_size: int,
    is_fixed: bool,
) -> chex.Array:
    """Pre-process PancakeSorting state for neural models.

    Args:
        current_stack: The stack array from the current state.
        target_stack: The stack array from the target state (None if fixed target).
        puzzle_size: The size of the pancake puzzle.
        is_fixed: Whether the puzzle has a fixed target.

    Returns:
        Normalized one-hot array in [-1, 1].
    """
    current_one_hot = jax.nn.one_hot(current_stack, num_classes=puzzle_size).flatten()
    if is_fixed:
        one_hots = current_one_hot
    else:
        target_one_hot = jax.nn.one_hot(target_stack, num_classes=puzzle_size).flatten()
        one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
    return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


def remove_face_centers(flatten_face: chex.Array, n: int) -> chex.Array:
    """Drop centre stickers from the flattened cube faces.

    Args:
        flatten_face: Flattened array of all face stickers.
        n: Cube dimension (e.g. 3 for a 3x3 cube).

    Returns:
        Flattened array with centre stickers removed (for odd n), or unchanged (for even n).
    """
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


def slidepuzzle_pre_process(
    current_board_unpacked: chex.Array,
    target_board_unpacked: chex.Array | None,
    size_square: int,
    is_fixed: bool,
) -> chex.Array:
    """Pre-process SlidePuzzle state for neural models.

    Args:
        current_board_unpacked: The unpacked board of the current state.
        target_board_unpacked: The unpacked board of the target state (None if fixed target).
        size_square: puzzle.size * puzzle.size.
        is_fixed: Whether the puzzle has a fixed target.

    Returns:
        Normalized one-hot array in [-1, 1].
    """
    current_board_one_hot = jax.nn.one_hot(current_board_unpacked, size_square).flatten()

    if is_fixed:
        one_hots = current_board_one_hot
    else:
        target_board_one_hot = jax.nn.one_hot(target_board_unpacked, size_square).flatten()
        one_hots = jnp.concatenate([target_board_one_hot, current_board_one_hot], axis=-1)

    return ((one_hots - 0.5) * 2.0).astype(DTYPE)


def rubikscube_face_one_hot(
    one_hot_fn,
    metric: str,
    puzzle_size: int,
    flatten_face: chex.Array,
) -> chex.Array:
    """Encode a single RubiksCube face array as a flat one-hot vector (un-normalized).

    Args:
        one_hot_fn: Callable that applies one-hot encoding to a face array.
        metric: The cube metric (e.g. "UQTM" uses all stickers; others drop centres).
        puzzle_size: The cube dimension (e.g. 3 for a 3x3 cube).
        flatten_face: Flattened face stickers array.

    Returns:
        Flat one-hot array (values in {0, 1}).
    """
    if metric == "UQTM":
        # UQTM needs all stickers including centres
        return one_hot_fn(flatten_face).flatten()
    else:
        face_no_centers = remove_face_centers(flatten_face, puzzle_size)
        # Create a one-hot encoding of the flattened face without centre stickers
        return one_hot_fn(face_no_centers).flatten()


def rubikscube_pre_process(
    one_hot_fn,
    metric: str,
    puzzle_size: int,
    current_flatten_face: chex.Array,
) -> chex.Array:
    """Pre-process a single RubiksCube face array into a normalized one-hot vector.

    Args:
        one_hot_fn: Callable that applies one-hot encoding to a face array.
        metric: The cube metric (e.g. "UQTM" or other).
        puzzle_size: The cube dimension (e.g. 3 for a 3x3 cube).
        current_flatten_face: Flattened face stickers array.

    Returns:
        Normalized one-hot array in [-1, 1].
    """
    one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, current_flatten_face)
    return ((one_hot - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


def rubikscube_random_pre_process(
    one_hot_fn,
    metric: str,
    puzzle_size: int,
    current_flatten_face: chex.Array,
    target_flatten_face: chex.Array,
) -> chex.Array:
    """Pre-process current and target RubiksCube face arrays for random-target models.

    Concatenates target and current one-hot encodings then normalizes.

    Args:
        one_hot_fn: Callable that applies one-hot encoding to a face array.
        metric: The cube metric (e.g. "UQTM" or other).
        puzzle_size: The cube dimension (e.g. 3 for a 3x3 cube).
        current_flatten_face: Flattened face stickers array for the current state.
        target_flatten_face: Flattened face stickers array for the target state.

    Returns:
        Normalized concatenated one-hot array in [-1, 1].
    """
    current_one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, current_flatten_face)
    target_one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, target_flatten_face)
    one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
    return ((one_hots - 0.5) * 2.0).astype(DTYPE)  # normalize to [-1, 1]


def slidepuzzle_zero_pos(board_unpacked: chex.Array) -> chex.Array:
    """Return a binary mask indicating the blank-tile position for SlidePuzzle conv models.

    Args:
        board_unpacked: The unpacked board of a SlidePuzzle state.

    Returns:
        Array of shape [size*size, 1] with 1.0 where the blank tile (0) is located.
    """
    return jnp.expand_dims(board_unpacked == 0, axis=-1).astype(jnp.float32)


def slidepuzzle_diff_pos(
    current_board_unpacked: chex.Array,
    target_board_unpacked: chex.Array,
    base_xy: chex.Array,
    puzzle_size: int,
) -> chex.Array:
    """Compute per-tile position difference for SlidePuzzle conv models.

    Args:
        current_board_unpacked: The unpacked board of the current state.
        target_board_unpacked: The unpacked board of the target state.
        base_xy: Base (x, y) coordinates for each tile position, shape [size*size, 2].
        puzzle_size: The edge length of the puzzle (e.g. 4 for a 4x4 puzzle).

    Returns:
        Array of shape [size*size, 2] with coordinate differences.
    """

    def to_xy(index):
        return index // puzzle_size, index % puzzle_size

    def pos(num, board):
        return to_xy(jnp.argmax(board == num))

    tpos = jnp.array(
        [pos(i, target_board_unpacked) for i in current_board_unpacked], dtype=jnp.int8
    )  # [size*size, 2]
    return base_xy - tpos


def preload_metadata(path, is_model_downloaded_fn, download_model_fn, load_params_fn):
    """Load metadata (and optionally cache params) from a saved model file.

    This logic is shared between NeuralHeuristicBase and NeuralQFunctionBase.

    Args:
        path: Path to the model file.
        is_model_downloaded_fn: Callable to check if model is downloaded.
        download_model_fn: Callable to download the model.
        load_params_fn: Callable to load params and metadata from a file.

    Returns:
        Tuple of (params_or_none, metadata_dict).
    """
    import pickle

    try:
        if not is_model_downloaded_fn(path):
            download_model_fn(path)
        params, metadata = load_params_fn(path)
        return params, metadata or {}
    except (FileNotFoundError, pickle.PickleError, OSError, RuntimeError) as e:
        print(f"Error loading metadata from {path}: {e}")
        return None, {}

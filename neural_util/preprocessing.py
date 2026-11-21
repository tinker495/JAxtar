import chex
import jax
import jax.numpy as jnp

from neural_util.dtypes import DTYPE


def lightsout_pre_process(
    current_board_unpacked: chex.Array,
    target_board_unpacked: chex.Array | None,
    is_fixed: bool,
) -> chex.Array:
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
    current_one_hot = jax.nn.one_hot(current_stack, num_classes=puzzle_size).flatten()
    if is_fixed:
        one_hots = current_one_hot
    else:
        target_one_hot = jax.nn.one_hot(target_stack, num_classes=puzzle_size).flatten()
        one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
    return ((one_hots - 0.5) * 2.0).astype(DTYPE)


def remove_face_centers(flatten_face: chex.Array, n: int) -> chex.Array:
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
    if metric == "UQTM":
        # UQTM needs all stickers including centres
        return one_hot_fn(flatten_face).flatten()
    else:
        face_no_centers = remove_face_centers(flatten_face, puzzle_size)
        return one_hot_fn(face_no_centers).flatten()


def rubikscube_pre_process(
    one_hot_fn,
    metric: str,
    puzzle_size: int,
    current_flatten_face: chex.Array,
) -> chex.Array:
    one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, current_flatten_face)
    return ((one_hot - 0.5) * 2.0).astype(DTYPE)


def rubikscube_random_pre_process(
    one_hot_fn,
    metric: str,
    puzzle_size: int,
    current_flatten_face: chex.Array,
    target_flatten_face: chex.Array,
) -> chex.Array:
    current_one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, current_flatten_face)
    target_one_hot = rubikscube_face_one_hot(one_hot_fn, metric, puzzle_size, target_flatten_face)
    one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
    return ((one_hots - 0.5) * 2.0).astype(DTYPE)


def slidepuzzle_zero_pos(board_unpacked: chex.Array) -> chex.Array:
    return jnp.expand_dims(board_unpacked == 0, axis=-1).astype(jnp.float32)


def slidepuzzle_diff_pos(
    current_board_unpacked: jax.Array,
    target_board_unpacked: jax.Array,
    base_xy: jax.Array,
    puzzle_size: int,
) -> jax.Array:
    """Return per-tile (row, col) displacement base_xy - target_pos with shape (size*size, 2)."""

    def to_xy(index):
        return index // puzzle_size, index % puzzle_size

    def pos(num, board):
        return to_xy(jnp.argmax(board == num))

    tpos = jnp.array(
        [pos(i, target_board_unpacked) for i in current_board_unpacked], dtype=jnp.int8
    )
    return base_xy - tpos


def preload_metadata(path, is_model_downloaded_fn, download_model_fn, load_params_fn):
    """Download if needed and load (params, metadata, resolved_path); returns (None, {}, path) on load failure."""
    import pickle

    try:
        resolved_path = path
        if not is_model_downloaded_fn(resolved_path):
            resolved_path = download_model_fn(resolved_path)
        params, metadata = load_params_fn(resolved_path)
        return params, metadata or {}, resolved_path
    except (FileNotFoundError, pickle.PickleError, OSError, RuntimeError) as e:
        print(f"Error loading metadata from {path}: {e}")
        return None, {}, path

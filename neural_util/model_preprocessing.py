from typing import Any

import chex
import jax
import jax.numpy as jnp

from neural_util.dtypes import DTYPE
from neural_util.preprocessing import (
    lightsout_pre_process,
    pancake_pre_process,
    rubikscube_pre_process,
    rubikscube_random_pre_process,
    slidepuzzle_diff_pos,
    slidepuzzle_pre_process,
    slidepuzzle_zero_pos,
)


class LightsOutPreProcessMixin:
    is_fixed: bool = True

    def pre_process(self, solve_config, current) -> chex.Array:
        target_board = solve_config.GoalSpec.board_unpacked
        return lightsout_pre_process(current.board_unpacked, target_board, self.is_fixed)


class LightsOutConvPreProcessMixin:
    network_model: Any = None

    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=self.network_model, **kwargs)

    def pre_process(self, solve_config, current) -> chex.Array:
        x = self.to_2d(self._diff(current, solve_config.GoalSpec))
        return ((x - 0.5) * 2.0).astype(DTYPE)

    def to_2d(self, x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (self.puzzle.size, self.puzzle.size, 1))

    def _diff(self, current, target) -> chex.Array:
        current_map = current.board_unpacked.astype(DTYPE)
        target_map = target.board_unpacked.astype(DTYPE)
        return jnp.not_equal(current_map, target_map).astype(DTYPE)


class PancakePreProcessMixin:
    is_fixed: bool = True

    def pre_process(self, solve_config, current) -> chex.Array:
        target_stack = solve_config.GoalSpec.stack
        return pancake_pre_process(current.stack, target_stack, self.puzzle.size, self.is_fixed)


class RubiksCubePreProcessMixin:
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)
        tile_count = puzzle.size * puzzle.size
        self._num_tile_classes = 6 if self._use_color_embedding else 6 * tile_count
        self.metric = puzzle.metric
        super().__init__(puzzle, **kwargs)

    def _one_hot_faces(self, faces: chex.Array) -> chex.Array:
        return jax.nn.one_hot(faces, num_classes=self._num_tile_classes)

    def pre_process(self, solve_config, current) -> chex.Array:
        current_flatten_face = current.faces_unpacked.flatten()
        return rubikscube_pre_process(
            self._one_hot_faces, self.metric, self.puzzle.size, current_flatten_face
        )


class RubiksCubeRandomPreProcessMixin(RubiksCubePreProcessMixin):
    is_fixed: bool = False

    def pre_process(self, solve_config, current) -> chex.Array:
        current_flatten_face = current.faces_unpacked.flatten()
        target_flatten_face = solve_config.GoalSpec.faces_unpacked.flatten()
        return rubikscube_random_pre_process(
            self._one_hot_faces,
            self.metric,
            self.puzzle.size,
            current_flatten_face,
            target_flatten_face,
        )


class SlidePuzzlePreProcessMixin:
    is_fixed: bool = True

    def __init__(self, puzzle, **kwargs):
        self.size_square = puzzle.size * puzzle.size
        super().__init__(puzzle, **kwargs)

    def pre_process(self, solve_config, current) -> chex.Array:
        return slidepuzzle_pre_process(
            current.board_unpacked,
            solve_config.GoalSpec.board_unpacked,
            self.size_square,
            self.is_fixed,
        )


class SlidePuzzleConvPreProcessMixin:
    base_xy: chex.Array
    network_model: Any = None

    def __init__(self, puzzle, **kwargs):
        self.size_square = puzzle.size * puzzle.size
        x = jnp.tile(jnp.arange(puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, puzzle.size, 1))
        y = jnp.tile(jnp.arange(puzzle.size)[jnp.newaxis, :, jnp.newaxis], (puzzle.size, 1, 1))
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)
        super().__init__(puzzle, model=self.network_model, **kwargs)

    def pre_process(self, solve_config, current) -> chex.Array:
        diff = self.to_2d(self._diff_pos(current, solve_config.GoalSpec))
        c_zero = self.to_2d(self._zero_pos(current))
        t_zero = self.to_2d(self._zero_pos(solve_config.GoalSpec))
        return jnp.concatenate([diff, c_zero, t_zero], axis=-1).astype(DTYPE)

    def to_2d(self, x: chex.Array) -> chex.Array:
        return x.reshape((self.puzzle.size, self.puzzle.size, x.shape[-1]))

    def _diff_pos(self, current, target) -> chex.Array:
        return slidepuzzle_diff_pos(
            current.board_unpacked, target.board_unpacked, self.base_xy, self.puzzle.size
        )

    def _zero_pos(self, current) -> chex.Array:
        return slidepuzzle_zero_pos(current.board_unpacked)


class SokobanPreProcessMixin:
    def pre_process(self, solve_config, current) -> chex.Array:
        target_board = solve_config.GoalSpec.board_unpacked
        current_board = current.board_unpacked
        stacked_board = jnp.concatenate([current_board, target_board], axis=-1)
        one_hot_board = jax.nn.one_hot(stacked_board, num_classes=4)
        flattened_board = jnp.reshape(one_hot_board, (-1,))
        return ((flattened_board - 0.5) * 2.0).astype(DTYPE)


class WorldModelPreProcessMixin:
    def pre_process(self, solve_config, current) -> chex.Array:
        target_latent = solve_config.GoalSpec.latent_unpacked.astype(jnp.float32)
        current_latent = current.latent_unpacked.astype(jnp.float32)
        latent_stack = jnp.concatenate([current_latent, target_latent], axis=-1)
        latent_stack = jnp.reshape(latent_stack, (-1,))
        return ((latent_stack - 0.5) * 2.0).astype(DTYPE)

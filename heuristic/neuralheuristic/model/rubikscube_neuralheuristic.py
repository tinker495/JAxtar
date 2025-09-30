from dataclasses import dataclass
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE


@dataclass(frozen=True)
class _CubieSpecs:
    corner_faces: chex.Array  # [8, 3]
    corner_rows: chex.Array  # [8, 3]
    corner_cols: chex.Array  # [8, 3]
    edge_faces: chex.Array  # [12, 2]
    edge_rows: chex.Array  # [12, 2]
    edge_cols: chex.Array  # [12, 2]


def _build_cubie_specs(size: int) -> Optional[_CubieSpecs]:
    """Return cube sticker indices grouped by cubie for size=3; otherwise None."""

    if size != 3:
        return None

    hi = size - 1
    mid = size // 2

    corner_faces = jnp.array(
        [
            [0, 3, 4],  # UFR
            [0, 3, 5],  # URB
            [0, 5, 2],  # UBL
            [0, 2, 4],  # ULF
            [1, 2, 4],  # DFL
            [1, 5, 2],  # DLB
            [1, 5, 3],  # DBR
            [1, 3, 4],  # DRF
        ],
        dtype=jnp.int32,
    )
    corner_rows = jnp.array(
        [
            [hi, 0, 0],  # UFR
            [0, 0, 0],
            [0, 0, 0],
            [hi, 0, 0],
            [0, hi, hi],
            [0, hi, hi],
            [hi, hi, hi],
            [hi, hi, hi],
        ],
        dtype=jnp.int32,
    )
    corner_cols = jnp.array(
        [
            [hi, 0, hi],  # UFR
            [hi, hi, 0],
            [0, hi, 0],
            [0, hi, 0],
            [0, hi, 0],
            [hi, 0, hi],
            [hi, hi, 0],
            [0, 0, hi],
        ],
        dtype=jnp.int32,
    )

    edge_faces = jnp.array(
        [
            [0, 4],  # UF
            [0, 3],  # UR
            [0, 5],  # UB
            [0, 2],  # UL
            [1, 4],  # DF
            [1, 3],  # DR
            [1, 5],  # DB
            [1, 2],  # DL
            [4, 3],  # FR
            [4, 2],  # FL
            [5, 3],  # BR
            [5, 2],  # BL
        ],
        dtype=jnp.int32,
    )
    edge_rows = jnp.array(
        [
            [hi, 0],  # UF
            [mid, 0],
            [0, 0],
            [mid, 0],
            [0, hi],
            [mid, hi],
            [hi, hi],
            [mid, hi],
            [mid, mid],
            [mid, mid],
            [mid, mid],
            [mid, mid],
        ],
        dtype=jnp.int32,
    )
    edge_cols = jnp.array(
        [
            [mid, mid],  # UF
            [hi, mid],
            [mid, mid],
            [0, mid],
            [mid, mid],
            [hi, mid],
            [mid, mid],
            [0, mid],
            [hi, 0],
            [0, hi],
            [0, hi],
            [hi, 0],
        ],
        dtype=jnp.int32,
    )

    return _CubieSpecs(
        corner_faces=corner_faces,
        corner_rows=corner_rows,
        corner_cols=corner_cols,
        edge_faces=edge_faces,
        edge_rows=edge_rows,
        edge_cols=edge_cols,
    )


class _RubiksCubeCubieNeuralHeuristicBase(NeuralHeuristicBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        self._cubie_specs = _build_cubie_specs(puzzle.size)
        super().__init__(puzzle, **kwargs)

    def _sticker_one_hot(self, state: RubiksCube.State) -> chex.Array:
        flattened = state.unpacked.faces.flatten()
        one_hot = jax.nn.one_hot(flattened, num_classes=6).flatten()
        return ((one_hot - 0.5) * 2.0).astype(DTYPE)

    def _cubie_encoding(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        specs = self._cubie_specs
        size = self.puzzle.size

        target_faces = solve_config.TargetState.unpacked.faces.reshape(6, size, size)
        current_faces = current.unpacked.faces.reshape(6, size, size)

        # Gather corner stickers for target and current states.
        def gather_corners(faces):
            return faces[specs.corner_faces, specs.corner_rows, specs.corner_cols]

        def gather_edges(faces):
            return faces[specs.edge_faces, specs.edge_rows, specs.edge_cols]

        target_corner_colors = gather_corners(target_faces)
        current_corner_colors = gather_corners(current_faces)
        target_edge_colors = gather_edges(target_faces)
        current_edge_colors = gather_edges(current_faces)

        # Corner orientation lookup: roll target colors to represent all three orientations.
        corner_lookup = jnp.stack(
            [
                target_corner_colors,
                jnp.roll(target_corner_colors, shift=-1, axis=1),
                jnp.roll(target_corner_colors, shift=-2, axis=1),
            ],
            axis=1,
        )  # [8, 3 orientations, 3]

        corner_matches = jnp.all(
            current_corner_colors[:, None, None, :] == corner_lookup[None, :, :, :],
            axis=-1,
        )  # [positions, cubies, orientations]

        # Edge orientation lookup: orientation 0 is reference, 1 is flipped.
        edge_lookup = jnp.stack(
            [target_edge_colors, target_edge_colors[:, ::-1]], axis=1
        )  # [12, 2, 2]
        edge_matches = jnp.all(
            current_edge_colors[:, None, None, :] == edge_lookup[None, :, :, :],
            axis=-1,
        )  # [positions, cubies, orientations]

        # Permutation one-hots: cubie -> position (corners: 8x8, edges: 12x12).
        corner_perm = corner_matches.any(axis=-1).astype(DTYPE).T.reshape(-1)
        edge_perm = edge_matches.any(axis=-1).astype(DTYPE).T.reshape(-1)

        # Orientation one-hots per cubie.
        corner_orient = corner_matches.transpose(1, 0, 2).astype(DTYPE).sum(axis=1).reshape(-1)
        edge_orient = edge_matches.transpose(1, 0, 2).astype(DTYPE).sum(axis=1).reshape(-1)

        features = jnp.concatenate([corner_perm, edge_perm, corner_orient, edge_orient], axis=0)
        return (features * 2.0 - 1.0).astype(DTYPE)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        if self._cubie_specs is None:
            # Fallback to sticker-wise encoding for unsupported cube sizes.
            target_vector = self._sticker_one_hot(solve_config.TargetState)
            current_vector = self._sticker_one_hot(current)
            if self.is_fixed:
                return current_vector
            return jnp.concatenate([target_vector, current_vector], axis=-1)

        return self._cubie_encoding(solve_config, current)


class RubiksCubeNeuralHeuristic(_RubiksCubeCubieNeuralHeuristicBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)


class RubiksCubeRandomNeuralHeuristic(_RubiksCubeCubieNeuralHeuristicBase):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

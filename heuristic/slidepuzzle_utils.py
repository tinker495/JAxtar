"""Shared utility functions for SlidePuzzle heuristic and Q-function."""

import chex
import jax
import jax.numpy as jnp
from puxle import SlidePuzzle


def diff_pos(
    current: SlidePuzzle.State,
    target: SlidePuzzle.State,
    puzzle_size: int,
    base_xy: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """Compute positional difference between current and target board states.

    Args:
        current: Current puzzle state.
        target: Target puzzle state.
        puzzle_size: Size of the puzzle grid.
        base_xy: Pre-computed coordinate grid of shape (size*size, 2).

    Returns:
        Tuple of (diff, tpos) where diff is (base_xy - tpos) and tpos
        holds the target positions for each tile in the current board.
    """

    def to_xy(index):
        return index // puzzle_size, index % puzzle_size

    def pos(num, board):
        return to_xy(jnp.argmax(board == num))

    tpos = jnp.array(
        [pos(i, target.board_unpacked) for i in current.board_unpacked], dtype=jnp.int8
    )
    diff = base_xy - tpos
    return (diff, tpos)


def linear_conflict(
    tpos: chex.Array,
    not_empty: chex.Array,
    diff: chex.Array,
    puzzle_size: int,
) -> int:
    """Compute the linear conflict count for the slide puzzle.

    Args:
        tpos: Target positions array of shape (size*size, 2).
        not_empty: Boolean mask of non-empty tiles, shape (size*size,).
        diff: Positional difference array of shape (size*size, 2).
        puzzle_size: Size of the puzzle grid.

    Returns:
        Total linear conflict value (each conflict adds 2 to distance).
    """
    tpos = jnp.reshape(tpos, (puzzle_size, puzzle_size, 2))
    not_empty = not_empty[:, jnp.newaxis]
    inrows = jnp.reshape(not_empty * (diff == 0), (puzzle_size, puzzle_size, 2))

    def _cond(val):
        _, _, conflict, _ = val
        return jnp.max(conflict) != 0

    def _while_count_conflict(val):
        pos, inrow, _, ans = val

        def _check_conflict(i, j):
            logic1 = i != j
            logic2 = jnp.logical_and(pos[i] > pos[j], i < j)
            logic3 = jnp.logical_and(pos[i] < pos[j], i > j)
            return jnp.logical_and(logic1, jnp.logical_or(logic2, logic3))

        i, j = jnp.arange(puzzle_size), jnp.arange(puzzle_size)
        i = jnp.expand_dims(i, axis=0)
        j = jnp.expand_dims(j, axis=1)
        conflict = jnp.sum(
            _check_conflict(i, j) * inrow[i] * inrow[j], axis=1, dtype=jnp.uint8
        )  # check conflict in rows

        max_idx = jnp.argmax(conflict)
        inrow = inrow.at[max_idx].set(False)
        ans += 1
        return pos, inrow, conflict, ans

    def _count_conflict(pos, inrow):
        _, _, _, conflict = jax.lax.while_loop(
            _cond,
            _while_count_conflict,
            (pos, inrow, jnp.ones(puzzle_size, dtype=jnp.uint8), -1),
        )
        return conflict * 2

    x_conflicts = jax.vmap(_count_conflict, in_axes=(1, 1))(tpos[:, :, 0], inrows[:, :, 1])
    y_conflicts = jax.vmap(_count_conflict, in_axes=(0, 0))(tpos[:, :, 1], inrows[:, :, 0])
    conflict = jnp.sum(x_conflicts) + jnp.sum(y_conflicts)
    return conflict

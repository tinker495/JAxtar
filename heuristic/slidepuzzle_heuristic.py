import chex
import jax
import jax.numpy as jnp
from puxle import SlidePuzzle

from heuristic.heuristic_base import Heuristic


class SlidePuzzleHeuristic(Heuristic):
    base_xy: chex.Array  # The coordinates of the numbers in the puzzle

    def __init__(self, puzzle: SlidePuzzle):
        super().__init__(puzzle)
        x = jnp.tile(
            jnp.arange(self.puzzle.size)[:, jnp.newaxis, jnp.newaxis], (1, self.puzzle.size, 1)
        )
        y = jnp.tile(
            jnp.arange(self.puzzle.size)[jnp.newaxis, :, jnp.newaxis], (self.puzzle.size, 1, 1)
        )
        self.base_xy = jnp.stack([x, y], axis=2).reshape(-1, 2)

    def distance(self, solve_config: SlidePuzzle.SolveConfig, current: SlidePuzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        current = current.unpacking()
        diff, tpos = self._diff_pos(current, solve_config.TargetState.unpacking())
        not_empty = current.board != 0
        return (
            self._manhattan_distance(not_empty, diff) + self._linear_conflict(tpos, not_empty, diff)
        ).astype(jnp.float32)

    def _diff_pos(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the difference between the state and the target.
        """

        def to_xy(index):
            return index // self.puzzle.size, index % self.puzzle.size

        def pos(num, board):
            return to_xy(jnp.argmax(board == num))

        tpos = jnp.array([pos(i, target.board) for i in current.board], dtype=jnp.int8)
        diff = self.base_xy - tpos
        return (diff, tpos)

    def _manhattan_distance(self, not_empty, diff) -> int:
        """
        This function should return the manhattan distance between the state and the target.
        """
        return jnp.sum(not_empty * jnp.sum(jnp.abs(diff), axis=1))

    def _linear_conflict(self, tpos, not_empty, diff) -> int:
        """
        This function should return the linear conflict between the state and the target.
        """
        tpos = jnp.reshape(tpos, (self.puzzle.size, self.puzzle.size, 2))
        not_empty = not_empty[:, jnp.newaxis]
        inrows = jnp.reshape(not_empty * (diff == 0), (self.puzzle.size, self.puzzle.size, 2))

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

            i, j = jnp.arange(self.puzzle.size), jnp.arange(self.puzzle.size)
            i = jnp.expand_dims(i, axis=0)
            j = jnp.expand_dims(j, axis=1)
            conflict = jnp.sum(
                _check_conflict(i, j) * inrow[i] * inrow[j], axis=1, dtype=jnp.uint8
            )  # check conflict in rows

            max_idx = jnp.argmax(conflict)
            inrow = inrow.at[max_idx].set(False)
            ans += 1
            # print(pos.shape, inrow.shape, conflict.shape, ans)
            return pos, inrow, conflict, ans

        def _count_conflict(pos, inrow):
            _, _, _, conflict = jax.lax.while_loop(
                _cond,
                _while_count_conflict,
                (pos, inrow, jnp.ones(self.puzzle.size, dtype=jnp.uint8), -1),
            )
            return conflict * 2

        x_conflicts = jax.vmap(_count_conflict, in_axes=(1, 1))(tpos[:, :, 0], inrows[:, :, 1])
        y_conflicts = jax.vmap(_count_conflict, in_axes=(0, 0))(tpos[:, :, 1], inrows[:, :, 0])
        conflict = jnp.sum(x_conflicts) + jnp.sum(y_conflicts)
        return conflict

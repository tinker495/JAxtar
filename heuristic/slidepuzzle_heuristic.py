

import chex
import jax
import jax.numpy as jnp

from puzzle.slidepuzzle import SlidePuzzle

class SlidePuzzleHeuristic:
    puzzle: SlidePuzzle

    def __init__(self, puzzle: SlidePuzzle):
        self.puzzle = puzzle

    def distance(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> int:
        """
        This function should return the distance between the state and the target.
        """
        diff, tpos = self._diff_pos(current, target)
        not_empty = (current.board != 0)
        return self._manhattan_distance(not_empty, diff) + self._linear_conflict(tpos, not_empty, diff)

        
    def _diff_pos(self, current: SlidePuzzle.State, target: SlidePuzzle.State) -> chex.Array:
        """
        This function should return the difference between the state and the target.
        """
        def to_xy(index):
            return index // self.puzzle.size, index % self.puzzle.size

        def pos(num, board):
            return to_xy(jnp.argmax(board == num))
        
        current_pos = jnp.array([pos(i, current.board) for i in range(0, self.puzzle.size ** 2)])
        target_pos = jnp.array([pos(i, target.board) for i in range(0, self.puzzle.size ** 2)])
        tpos = jnp.array([pos(i, target.board) for i in current.board], dtype=jnp.int8)
        num_diff = current_pos - target_pos
        return (jnp.take_along_axis(num_diff, jnp.expand_dims(current.board,axis=1), axis=0),
                tpos)


    def _manhattan_distance(self, not_empty, diff) -> int:
        """
        This function should return the manhattan distance between the state and the target.
        """
        return jnp.sum(not_empty * jnp.sum(jnp.abs(diff),axis=1))
    
    def _linear_conflict(self, tpos, not_empty, diff) -> int:
        """
        This function should return the linear conflict between the state and the target.
        """
        tpos = jnp.reshape(tpos, (self.puzzle.size, self.puzzle.size, 2))
        not_empty = jnp.expand_dims(not_empty, axis=1)
        inrows = jnp.reshape(not_empty * (diff == 0), (self.puzzle.size, self.puzzle.size, 2))

        def _cond(val):
            _, _, conflict, _ = val
            return jnp.max(conflict) != 0

        def _while_count_conflict(val):
            pos, inrow, _, ans = val
            def _check_conflict(i,j):
                logic1 = i != j
                logic2 = jnp.logical_and(pos[i] > pos[j],i < j)
                logic3 = jnp.logical_and(pos[i] < pos[j],i > j)
                return jnp.logical_and(logic1,jnp.logical_or(logic2, logic3))
            i, j = jnp.arange(self.puzzle.size), jnp.arange(self.puzzle.size)
            i = jnp.expand_dims(i, axis=0)
            j = jnp.expand_dims(j, axis=1)
            conflict = jnp.sum(_check_conflict(i,j) * inrow[i] * inrow[j], axis=1, dtype=jnp.uint8) # check conflict in rows
            
            max_idx = jnp.argmax(conflict)
            inrow = inrow.at[max_idx].set(False)
            ans += 1
            #print(pos.shape, inrow.shape, conflict.shape, ans)
            return pos, inrow, conflict, ans
        
        def _count_conflict(pos, inrow):
            _, _, _, conflict = jax.lax.while_loop(_cond, _while_count_conflict, (pos, inrow, jnp.ones(self.puzzle.size, dtype=jnp.uint8), -1))
            return conflict * 2
        
        x_conflicts = jax.vmap(_count_conflict, in_axes=(1,1))(tpos[:,:,0], inrows[:,:,0])
        y_conflicts = jax.vmap(_count_conflict, in_axes=(0,0))(tpos[:,:,1], inrows[:,:,1])
        conflict = jnp.sum(x_conflicts) + jnp.sum(y_conflicts)
        return conflict
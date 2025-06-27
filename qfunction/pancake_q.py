from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from puxle import PancakeSorting

from qfunction.q_base import QFunction


class PancakeQ(QFunction):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def q_value(
        self,
        solve_config: PancakeSorting.SolveConfig,
        current: PancakeSorting.State,
        params: Optional[Any] = None,
    ) -> chex.Array:
        """
        Get q values for all possible actions from the current state.
        For EmptyQFunction, this returns an array of zeros for each available move.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        dists = jax.vmap(self._distance, in_axes=(None, 0))(solve_config, neighbors)
        return dists

    def _distance(
        self, solve_config: PancakeSorting.SolveConfig, current: PancakeSorting.State
    ) -> float:
        """
        Gap heuristic for pancake sorting.

        This heuristic counts the number of adjacent pancakes that are out of order.
        Each flip operation can at most fix one adjacency relationship, so this is
        an admissible heuristic (it never overestimates the true distance).

        Returns:
            float: Estimated number of flip operations needed to reach the target state
        """
        current_stack = current.stack
        target_stack = solve_config.TargetState.stack

        # Count the gaps (adjacent pancakes that are not in the correct relative order)
        # In a sorted stack, each pancake should be 1 smaller than the one below it
        # Create arrays of adjacent pancakes
        top_pancakes = current_stack[:-1]
        bottom_pancakes = current_stack[1:]

        # Calculate where there are gaps (where top + 1 != bottom)
        # This returns a boolean array where True indicates a gap
        gaps_mask = (top_pancakes + 1) != bottom_pancakes

        # Sum the mask to get the number of gaps
        gaps = jnp.sum(gaps_mask)

        # Count misplaced pancakes (not in correct position)
        misplaced_mask = current_stack != target_stack
        misplaced = jnp.sum(misplaced_mask)

        # Return the maximum of the two heuristics
        # Divide misplaced by 2 as an optimization since one flip can fix multiple misplacements
        return jnp.maximum(gaps, misplaced / 2.0)

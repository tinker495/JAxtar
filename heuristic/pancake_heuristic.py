import jax
import jax.numpy as jnp
from puxle import PancakeSorting

from heuristic.heuristic_base import Heuristic


class PancakeHeuristic(Heuristic):
    def __init__(self, puzzle):
        super().__init__(puzzle)

    def distance(
        self, solve_config: PancakeSorting.SolveConfig, current: PancakeSorting.State
    ) -> jax.Array:
        """Gap heuristic (admissible because one flip fixes at most one adjacency), combined with misplaced / 2."""
        current_stack = current.stack
        target_stack = solve_config.GoalSpec.stack

        # Sorted stack increases by 1 from top to bottom, so top + 1 != bottom marks a gap.
        top_pancakes = current_stack[:-1]
        bottom_pancakes = current_stack[1:]

        gaps_mask = (top_pancakes + 1) != bottom_pancakes

        gaps = jnp.sum(gaps_mask)

        misplaced_mask = current_stack != target_stack
        misplaced = jnp.sum(misplaced_mask)

        # Scale misplaced count by 1/2 because a single flip can reposition multiple pancakes.
        return jnp.maximum(gaps, misplaced / 2.0)

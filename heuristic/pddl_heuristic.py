import jax.numpy as jnp
from puxle import PDDL

from heuristic.heuristic_base import Heuristic


class PDDLHeuristic(Heuristic):
    def __init__(self, puzzle: PDDL):
        super().__init__(puzzle)

    def distance(self, solve_config: PDDL.SolveConfig, current: PDDL.State) -> float:
        """
        Simple admissible heuristic for PDDL: number of goal atoms not yet satisfied.
        h(s) = |{ g in GoalMask | g is False in s }|
        """
        atoms = current.unpacked_atoms
        goal_mask = solve_config.GoalMask
        missing = jnp.logical_and(goal_mask, jnp.logical_not(atoms))
        return jnp.sum(missing).astype(jnp.float32)

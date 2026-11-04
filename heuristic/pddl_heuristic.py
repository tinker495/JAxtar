import jax
import jax.numpy as jnp
from puxle import PDDL

from heuristic.heuristic_base import Heuristic
from puxle.pddls.pddl import PDDLSolveConfig
from puxle.pddls.state_defs import PDDLGoalSpec, PDDLInstanceContext, PDDLState


class PDDLHeuristic(Heuristic[PDDLState, PDDLInstanceContext, PDDLGoalSpec]):
    def __init__(self, puzzle: PDDL):
        super().__init__(puzzle)

    def distance(self, solve_config: PDDLSolveConfig, current: PDDLState) -> jax.Array:
        """
        Simple admissible heuristic for PDDL: number of goal atoms not yet satisfied.
        h(s) = |{ g in GoalMask | g is False in s }|
        """
        atoms = current.unpacked_atoms
        goal_mask = solve_config.GoalSpec.GoalMask
        missing = jnp.logical_and(goal_mask, jnp.logical_not(atoms))
        return jnp.sum(missing).astype(jnp.float32)

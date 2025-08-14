import jax
import jax.numpy as jnp
from puxle import PDDL

from qfunction.q_base import QFunction


class PDDLQ(QFunction):
    def __init__(self, puzzle: PDDL):
        super().__init__(puzzle)

    def q_value(self, solve_config: PDDL.SolveConfig, current: PDDL.State):
        """
        Q-values for PDDL are computed as the heuristic value of each neighbor:
        number of unsatisfied goal atoms in the neighbor state.
        """
        neighbors, _ = self.puzzle.get_neighbours(solve_config, current)
        return jax.vmap(self._distance, in_axes=(None, 0))(solve_config, neighbors)

    def _distance(self, solve_config: PDDL.SolveConfig, state: PDDL.State) -> float:
        atoms = state.unpacked_atoms
        goal_mask = solve_config.GoalMask
        missing = jnp.logical_and(goal_mask, jnp.logical_not(atoms))
        return jnp.sum(missing).astype(jnp.float32)

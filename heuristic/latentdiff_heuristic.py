import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def get_distance(A: chex.Array, B: chex.Array):
    return jnp.sum(A * B, axis=-1)


class LatentDiffHeuristic(Heuristic):
    def __init__(self, puzzle: WorldModelPuzzleBase):
        super().__init__(puzzle)

    def batched_distance(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> chex.Array:
        """
        This function should return the distance between the state and the target.
        """
        current_latents = jax.vmap(self.puzzle.from_uint8)(current.latent)

        target_projected_latents = solve_config.projected_latent[jnp.newaxis, ...]

        current_projected_latents = self.puzzle.model.apply(
            self.puzzle.params,
            current_latents,
            training=False,
            method=self.puzzle.model.forward_project,
        )

        distance = get_distance(target_projected_latents, current_projected_latents)
        return distance

    def distance(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return self.batched_distance(solve_config, current[jnp.newaxis, ...])[0]

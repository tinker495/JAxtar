import chex
import jax
import jax.numpy as jnp

from heuristic.heuristic_base import Heuristic
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


def similarity_loss_fn(A: chex.Array, B: chex.Array):
    A_norm = jnp.sqrt(jnp.sum(A**2, axis=-1, keepdims=True) + 1e-8)
    B_norm = jnp.sqrt(jnp.sum(B**2, axis=-1, keepdims=True) + 1e-8)
    dot_product = jnp.sum(A * B, axis=-1, keepdims=True)
    similarity = dot_product / (A_norm * B_norm)
    return 1.0 - similarity


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
            self.puzzle.params, current_latents, training=False, method=self.puzzle.model.project
        )

        similarity_loss = similarity_loss_fn(target_projected_latents, current_projected_latents)[
            :, 0
        ]
        return similarity_loss * 100

    def distance(
        self, solve_config: WorldModelPuzzleBase.SolveConfig, current: WorldModelPuzzleBase.State
    ) -> float:
        """
        Return zero distance for any puzzle state.
        """
        return self.batched_distance(solve_config, current[jnp.newaxis, ...])[0]

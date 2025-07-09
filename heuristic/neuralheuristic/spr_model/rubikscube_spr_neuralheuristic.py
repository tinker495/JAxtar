import chex
import jax
import jax.numpy as jnp
from puxle import RubiksCube

from heuristic.neuralheuristic.spr_neuralheuristic_base import SPRNeuralHeuristic
from neural_util.modules import DTYPE


class RubiksCubeSPRNeuralHeuristic(SPRNeuralHeuristic):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_flatten_face = current.unpacked.faces.flatten()
        current_one_hot = jax.nn.one_hot(current_flatten_face, num_classes=6).flatten()
        if self.is_fixed:
            one_hots = current_one_hot
        else:
            target_flatten_face = solve_config.TargetState.unpacked.faces.flatten()
            target_one_hot = jax.nn.one_hot(target_flatten_face, num_classes=6).flatten()
            one_hots = jnp.concatenate([target_one_hot, current_one_hot], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)

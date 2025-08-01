import chex
import jax.numpy as jnp
from flax import linen as nn
from puxle import LightsOut

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    ConvResBlock,
    ResBlock,
    conditional_dummy_norm,
)


class LightsOutNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: LightsOut, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: LightsOut.SolveConfig, current: LightsOut.State
    ) -> chex.Array:
        current_map = current.unpacked.board.astype(DTYPE)
        if self.is_fixed:
            one_hots = current_map
        else:
            target_map = solve_config.TargetState.unpacked.board.astype(DTYPE)
            one_hots = jnp.concatenate([target_map, current_map], axis=-1)
        return ((one_hots - 0.5) * 2.0).astype(DTYPE)


class Model(nn.Module):
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x, training=False):
        # [4, 4, 1] -> conv
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = ConvResBlock(64, (3, 3), strides=1, norm_fn=self.norm_fn)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = ResBlock(512, norm_fn=self.norm_fn)(x, training)
        x = nn.Dense(1, dtype=DTYPE)(x)
        _ = conditional_dummy_norm(x, self.norm_fn, training)
        return x


class LightsOutConvNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: LightsOut, **kwargs):
        super().__init__(puzzle, model=Model, **kwargs)

    def pre_process(
        self, solve_config: LightsOut.SolveConfig, current: LightsOut.State
    ) -> chex.Array:
        x = self.to_2d(self._diff(current, solve_config.TargetState))
        return ((x - 0.5) * 2.0).astype(DTYPE)

    def to_2d(self, x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (self.puzzle.size, self.puzzle.size, 1))

    def _diff(self, current: LightsOut.State, target: LightsOut.State) -> chex.Array:
        """
        This function should return the difference, not_equal of the current state and the target state
        """
        current_map = current.unpacked.board.astype(DTYPE)
        target_map = target.unpacked.board.astype(DTYPE)
        return jnp.not_equal(current_map, target_map).astype(DTYPE)

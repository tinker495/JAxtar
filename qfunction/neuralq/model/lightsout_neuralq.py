import chex
import jax.numpy as jnp
from flax import linen as nn
from puxle import LightsOut

from neural_util.basemodel import DistanceModel
from neural_util.dtypes import DTYPE, PARAM_DTYPE
from neural_util.modules import DEFAULT_NORM_FN, ConvResBlock, ResBlock, apply_norm
from neural_util.preprocessing import lightsout_pre_process
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class LightsOutNeuralQ(NeuralQFunctionBase):
    is_fixed: bool = True

    def __init__(self, puzzle: LightsOut, **kwargs):
        super().__init__(puzzle, **kwargs)

    def pre_process(
        self, solve_config: LightsOut.SolveConfig, current: LightsOut.State
    ) -> chex.Array:
        target_board = solve_config.TargetState.board_unpacked
        return lightsout_pre_process(current.board_unpacked, target_board, self.is_fixed)


class LightsOutRandomNeuralQ(LightsOutNeuralQ):
    is_fixed: bool = False


class Model(DistanceModel):
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x, training=False):
        # [4, 4, 1] -> conv
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ConvResBlock(64, (3, 3), strides=1)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(1024, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ResBlock(1024)(x, training)
        x = nn.Dense(self.action_size, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        return x


class LightsOutConvNeuralQ(NeuralQFunctionBase):
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
        current_map = current.board_unpacked.astype(DTYPE)
        target_map = target.board_unpacked.astype(DTYPE)
        return jnp.not_equal(current_map, target_map).astype(jnp.float32)

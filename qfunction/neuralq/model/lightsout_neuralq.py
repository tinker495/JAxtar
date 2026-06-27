import jax.numpy as jnp
from flax import linen as nn

from neural_util.basemodel import DistanceModel
from neural_util.dtypes import DTYPE, PARAM_DTYPE
from neural_util.model_preprocessing import LightsOutConvPreProcessMixin, LightsOutPreProcessMixin
from neural_util.modules import DEFAULT_NORM_FN, ConvResBlock, ResBlock, apply_norm
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class LightsOutNeuralQ(LightsOutPreProcessMixin, NeuralQFunctionBase):
    is_fixed: bool = True


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


class LightsOutConvNeuralQ(LightsOutConvPreProcessMixin, NeuralQFunctionBase):
    network_model = Model

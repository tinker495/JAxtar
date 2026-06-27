import jax.numpy as jnp
from flax import linen as nn

from neural_util.basemodel import DistanceModel
from neural_util.dtypes import DTYPE, PARAM_DTYPE
from neural_util.modules import DEFAULT_NORM_FN, ConvResBlock, ResBlock, apply_norm


class SlidePuzzleConvModel(DistanceModel):
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Conv(256, (3, 3), strides=1, padding="SAME", dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ConvResBlock(256, (3, 3), strides=1, norm_fn=self.norm_fn)(x, training)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(512, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = nn.relu(x)
        x = ResBlock(512, norm_fn=self.norm_fn)(x, training)
        return nn.Dense(self.action_size, dtype=DTYPE, param_dtype=PARAM_DTYPE)(x)

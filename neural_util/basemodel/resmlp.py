from flax import linen as nn

from neural_util.basemodel.base import DistanceModel
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
)


class ResMLPModel(DistanceModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0

    @nn.compact
    def __call__(self, x, training=False):
        if self.use_swiglu:
            x = Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE)(x, training)
            if self.resblock_fn != PreActivationResBlock:
                x = Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE)(x, training)
            else:
                x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
        else:
            x = nn.Dense(self.initial_dim, dtype=DTYPE)(x)
            x = self.norm_fn(x, training, dtype=DTYPE)
            x = self.activation(x)
            x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
            if self.resblock_fn != PreActivationResBlock:
                x = self.norm_fn(x, training, dtype=DTYPE)
                x = self.activation(x)
        for _ in range(self.Res_N - self.tail_head_precision):
            x = self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
            )(x, training)
        for _ in range(self.tail_head_precision):
            x = self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=HEAD_DTYPE,
                param_dtype=HEAD_DTYPE,
            )(x, training)
        if self.resblock_fn == PreActivationResBlock:
            x = self.norm_fn(x, training, dtype=HEAD_DTYPE)
            x = self.activation(x)
        x = x.astype(HEAD_DTYPE)
        x = nn.Dense(
            self.action_size, dtype=HEAD_DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        return x

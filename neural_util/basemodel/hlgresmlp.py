import jax.numpy as jnp
from flax import linen as nn

from neural_util.basemodel.base import DistanceHLGModel
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
)


class MLP(nn.Module):

    hidden_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
        x = self.norm_fn(x, training, dtype=DTYPE)
        x = self.activation(x)
        return x


class preactivation_MLP(nn.Module):
    hidden_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        x = self.norm_fn(x, training, dtype=self.dtype)
        x = self.activation(x)
        x = nn.Dense(
            self.hidden_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        return x


class HLGResMLPModel(DistanceHLGModel):
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

    def setup(self):
        super().setup()

        self.initial_mlp = (
            Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE)
            if self.use_swiglu
            else MLP(self.initial_dim, norm_fn=self.norm_fn, activation=self.activation)
        )
        self.second_mlp = (
            (
                Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE)
                if self.use_swiglu
                else MLP(self.hidden_dim, norm_fn=self.norm_fn, activation=self.activation)
            )
            if self.resblock_fn != PreActivationResBlock
            else nn.Dense(self.hidden_dim, dtype=DTYPE)
        )

        self.resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
            )
            for _ in range(self.Res_N - self.tail_head_precision)
        ]
        self.tail_head_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=HEAD_DTYPE,
                param_dtype=HEAD_DTYPE,
            )
            for _ in range(self.tail_head_precision)
        ]
        self.final_dense = (
            preactivation_MLP(self.action_size * self.categorial_n, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size * self.categorial_n,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
            )
        )

    def get_logits(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        for resblock in self.resblocks:
            x = resblock(x, training)
        for resblock in self.tail_head_resblocks:
            x = resblock(x, training)

        if isinstance(self.final_dense, nn.Dense):
            x = x.astype(HEAD_DTYPE)
            x = self.final_dense(x)
        else:
            x = self.final_dense(x, training)

        x = x.reshape(x.shape[:-1] + (self.action_size, self.categorial_n))
        return x

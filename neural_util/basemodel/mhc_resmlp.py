import jax.numpy as jnp
from flax import linen as nn

from neural_util.basemodel.base import DistanceModel
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    MLP,
    MHCLayer,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
    preactivation_MLP,
)


class MHCResMLPModel(DistanceModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0
    n_streams: int = 4
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01
    rms_norm_epsilon: float = 1e-6

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

        mhc_layers = []
        for _ in range(self.Res_N - self.tail_head_precision):
            mhc_layers.append(
                MHCLayer(
                    self.hidden_dim * self.hidden_node_multiplier,
                    hidden_N=self.hidden_N,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    use_swiglu=self.use_swiglu,
                    n_streams=self.n_streams,
                    sinkhorn_iters=self.sinkhorn_iters,
                    alpha_init=self.alpha_init,
                    rms_norm_epsilon=self.rms_norm_epsilon,
                )
            )
        for _ in range(self.tail_head_precision):
            mhc_layers.append(
                MHCLayer(
                    self.hidden_dim * self.hidden_node_multiplier,
                    hidden_N=self.hidden_N,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    use_swiglu=self.use_swiglu,
                    n_streams=self.n_streams,
                    sinkhorn_iters=self.sinkhorn_iters,
                    alpha_init=self.alpha_init,
                    rms_norm_epsilon=self.rms_norm_epsilon,
                    dtype=HEAD_DTYPE,
                    param_dtype=HEAD_DTYPE,
                )
            )
        self.mhc_layers = tuple(mhc_layers)

        self.final_dense = (
            preactivation_MLP(self.action_size, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
            )
        )

    def __call__(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        x_stream = jnp.repeat(x[:, None, :], repeats=self.n_streams, axis=1)
        for layer in self.mhc_layers:
            x_stream = layer(x_stream, training)

        x = jnp.mean(x_stream, axis=1)
        if isinstance(self.final_dense, nn.Dense):
            x = x.astype(HEAD_DTYPE)
            x = self.final_dense(x)
        else:
            x = self.final_dense(x, training)
        return x

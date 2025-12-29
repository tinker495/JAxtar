import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
)
from train_util.losses import loss_from_diff


class BaseModel(nn.Module):
    output_dim: int = 1
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
            self.output_dim, dtype=HEAD_DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        return x

    def train_loss(
        self, x, target, actions=None, loss_type="mse", loss_args=None, td_error_clip=None, **kwargs
    ):
        pred = self(x, training=True)
        if actions is not None:
            pred = jnp.take_along_axis(pred, actions[:, jnp.newaxis], axis=1)

        diff = target - pred

        if td_error_clip is not None and td_error_clip > 0:
            clip_val = jnp.asarray(td_error_clip, dtype=diff.dtype)
            diff = jnp.clip(diff, -clip_val, clip_val)

        return loss_from_diff(diff, loss=loss_type, loss_args=loss_args)


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


class BaseHLGModel(nn.Module):
    output_dim: int = 1
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

    categorial_n: int = 100
    vmin: float = -1.0
    vmax: float = 30.0
    _sigma: float = 0.75

    def setup(self):
        self.categorial_bins = np.linspace(
            self.vmin, self.vmax, self.categorial_n + 1
        )  # (categorial_n + 1,)
        self.categorial_centers = (
            self.categorial_bins[:-1] + self.categorial_bins[1:]
        ) / 2  # (categorial_n,)
        self.categorial_centers = self.categorial_centers.reshape(1, 1, -1)  # (1, 1, categorial_n)
        self.sigma = self._sigma * (self.categorial_bins[1] - self.categorial_bins[0])

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
            preactivation_MLP(self.output_dim * self.categorial_n, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.output_dim * self.categorial_n,
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

        x = x.reshape(x.shape[0], self.output_dim, self.categorial_n)
        return x

    def __call__(self, x, training=False):
        logits = self.get_logits(x, training)
        softmax = jax.nn.softmax(logits, axis=-1)
        categorial_centers = self.categorial_centers
        x = jnp.sum(softmax * categorial_centers, axis=-1)  # (batch_size, output_dim)
        return x

    def train_loss(self, x, target_q, actions=None, **kwargs):
        categorial_bins, sigma = self.categorial_bins, self.sigma
        # target: [batch, 1]

        def f(target):
            cdf_evals = jax.scipy.special.erf((categorial_bins - target) / (jnp.sqrt(2) * sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(f)(target_q)
        logits_actions = self.get_logits(x, training=True)
        if actions is None:
            logits = logits_actions.squeeze(1)
        else:
            logits = jnp.take_along_axis(
                logits_actions, actions[:, jnp.newaxis, jnp.newaxis], axis=1
            ).squeeze(1)
        sce = optax.softmax_cross_entropy(logits, target_probs)  # (batch_size,)
        return sce

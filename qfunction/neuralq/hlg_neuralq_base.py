import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from neural_util.modules import DEFAULT_NORM_FN, DTYPE, ResBlock, conditional_dummy_norm
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class HLGQModelBase(nn.Module):
    action_size: int = 4
    categorial_n: int = 100
    vmin: float = -1.0
    vmax: float = 30.0
    _sigma: float = 0.75
    Res_N: int = 4
    hidden_N: int = 1
    hidden_dim: int = 1000
    activation: str = nn.relu
    norm_fn: callable = DEFAULT_NORM_FN

    def setup(self):
        self.categorial_bins = jnp.linspace(
            self.vmin, self.vmax, self.categorial_n + 1
        )  # (categorial_n + 1,)
        self.categorial_centers = (
            self.categorial_bins[:-1] + self.categorial_bins[1:]
        ) / 2  # (categorial_n,)
        self.categorial_centers = self.categorial_centers.reshape(1, 1, -1)  # (1, 1, categorial_n)
        self.sigma = self._sigma * (self.categorial_bins[1] - self.categorial_bins[0])

        self.input_layer = nn.Dense(5000, dtype=DTYPE)
        self.hidden_layer = nn.Dense(self.hidden_dim, dtype=DTYPE)
        self.res_blocks = [
            ResBlock(
                self.hidden_dim,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
            )
            for _ in range(self.Res_N)
        ]
        self.output_layer = nn.Dense(
            self.action_size * self.categorial_n,
            dtype=DTYPE,
            kernel_init=nn.initializers.normal(stddev=0.01),
        )
        self.dummy_norm = conditional_dummy_norm(self.output_layer, self.norm_fn)

    def __call__(self, x, training=False):
        x = self.input_layer(x)
        x = self.norm_fn(x, training)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.norm_fn(x, training)
        x = self.activation(x)
        for res_block in self.res_blocks:
            x = res_block(x, training)
        x = self.output_layer(x)
        _ = self.dummy_norm(x, training)
        logits = x.reshape(
            x.shape[0], self.action_size, self.categorial_n
        )  # (batch_size, action_size, categorial_n)
        probs = jax.nn.softmax(logits, axis=-1)  # (batch_size, action_size, categorial_n)
        q = jnp.sum(probs * self.categorial_centers, axis=-1)  # (batch_size, action_size)
        return logits, q

    def train(self, x, actions, target_q):
        # target: [batch, 1]
        def f(target):
            cdf_evals = jax.scipy.special.erf(
                (self.categorial_bins - target) / (jnp.sqrt(2) * self.sigma)
            )
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(f)(target_q)
        logits_actions, _ = self(x, training=True)
        logits = jnp.take_along_axis(
            logits_actions, actions[:, jnp.newaxis, jnp.newaxis], axis=1
        ).squeeze(
            1
        )  # (batch_size, categorial_n)
        sce = optax.softmax_cross_entropy(logits, target_probs)  # (batch_size,)
        return sce


class HLGNeuralQFunctionBase(NeuralQFunctionBase):
    def __init__(
        self, puzzle, model=HLGQModelBase, categorial_n=100, vmin=-1.0, vmax=30.0, **kwargs
    ):
        super().__init__(
            puzzle, model=model, categorial_n=categorial_n, vmin=vmin, vmax=vmax, **kwargs
        )

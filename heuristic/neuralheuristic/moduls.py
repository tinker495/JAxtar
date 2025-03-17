import flax.linen as nn
import jax
import jax.numpy as jnp


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)


def hl_gaussian_convert(target, support, sigma):
    def f(target):
        cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
        z = cdf_evals[-1] - cdf_evals[0]
        bin_probs = cdf_evals[1:] - cdf_evals[:-1]
        return bin_probs / z

    return jax.vmap(f)(target)


def first_high_zero_init(key, shape, dtype=jnp.float32, scale=10.0):
    # When categorical initialization is random, it would have the middle value of categorical size.
    # We want it to predict 0 at the beginning, so we initialize the first weight to a high value
    # and all others to zero.
    zeros = jnp.full(shape, -scale, dtype=dtype)
    zeros = zeros.at[0].set(scale)
    return zeros


class CategorialOutput(nn.Module):
    max_distance: int

    def setup(self):
        self.support_size = self.max_distance + 1
        self.support = jnp.arange(self.support_size)

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.support_size, bias_init=first_high_zero_init)(x)
        probs = nn.softmax(logits)
        mul = probs * self.support
        scalar = jnp.sum(mul, axis=1)
        return probs, scalar

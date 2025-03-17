from functools import partial

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


def first_high_zero_init(key, shape, dtype=jnp.float32, scale=10.0, action_size=4, support_size=30):
    # When categorical initialization is random, it would have the middle value of categorical size.
    # We want it to predict 0 at the beginning, so we initialize the first weight to a high value
    # and all others to zero.
    # For each action, we set a specific index to high value: 0, action_size-1, 2*(action_size-1), etc.
    zeros = jnp.full(shape, -scale, dtype=dtype)
    indices = jnp.arange(action_size) * support_size
    zeros = zeros.at[indices].set(scale)
    return zeros


class CategorialOutput(nn.Module):
    action_size: int
    max_distance: int

    def setup(self):
        self.support_size = self.max_distance + 1
        self.support = jnp.arange(self.support_size)

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(
            self.action_size * self.support_size,
            bias_init=partial(
                first_high_zero_init, action_size=self.action_size, support_size=self.support_size
            ),
        )(x)
        logits = jnp.reshape(logits, (-1, self.action_size, self.support_size))  # [B, A, H]
        probs = nn.softmax(logits, axis=2)  # [B, A, H]
        mul = probs * self.support  # [B, A, H]
        scalar = jnp.sum(mul, axis=2)  # [B, A]
        return probs, scalar

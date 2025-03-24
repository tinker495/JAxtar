from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


def cosine_similarity(x, y):
    return jnp.einsum("bd, bd -> b", x, y) / (
        jnp.linalg.norm(x, axis=1) * jnp.linalg.norm(y, axis=1) + 1e-6
    )


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)


def LayerNorm(x, training):
    return nn.LayerNorm()(x)


# Residual Block
class ResBlock(nn.Module):
    node_size: int
    norm_fn: Callable = LayerNorm

    @nn.compact
    def __call__(self, x0, training=False):
        x = self.norm_fn(x0, training)
        x = nn.Dense(self.node_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        return x + x0


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    norm_fn: Callable = LayerNorm

    @nn.compact
    def __call__(self, x0, training=False):
        x = self.norm_fn(x0, training)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        return x + x0

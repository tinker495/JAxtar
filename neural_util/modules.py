from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

DTYPE = jnp.bfloat16


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.99, dtype=DTYPE)(x, use_running_average=not training)


def LayerNorm(x, training):
    return nn.LayerNorm()(x)


# Residual Block
class ResBlock(nn.Module):
    node_size: int
    norm_fn: Callable = BatchNorm

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size)(x0)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        x = self.norm_fn(x, training)
        return nn.relu(x + x0)


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    norm_fn: Callable = BatchNorm

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x0)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = nn.Conv(self.filters, self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = self.norm_fn(x, training)
        return nn.relu(x + x0)

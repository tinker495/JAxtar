from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .norm import BatchReNorm as BatchReNorm_

DTYPE = jnp.bfloat16


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9, dtype=DTYPE)(x, use_running_average=not training)


def BatchReNorm(x, training):
    return BatchReNorm_(momentum=0.99, dtype=DTYPE)(x, use_running_average=not training)


def InstanceNorm(x, training):
    return nn.InstanceNorm(dtype=DTYPE)(x)


def LayerNorm(x, training):
    return nn.LayerNorm(dtype=DTYPE)(x)


def GroupNorm(x, training):
    return nn.GroupNorm(num_groups=10, dtype=DTYPE)(x)


def RMSNorm(x, training):
    return nn.RMSNorm(dtype=DTYPE)(x)


DEFAULT_NORM_FN = BatchNorm


def conditional_dummy_norm(x, norm_fn, training):
    if norm_fn != BatchNorm and norm_fn != BatchReNorm:
        return norm_fn(x, training)
    else:
        return x


# Norm function registry for config-driven selection
NORM_FN_REGISTRY = {
    "batch": BatchNorm,
    "batchrenorm": BatchReNorm,
    "instance": InstanceNorm,
    "layer": LayerNorm,
    "group": GroupNorm,
    "rms": RMSNorm,
}


def get_norm_fn(norm_name_or_fn=None):
    if norm_name_or_fn is None:
        return DEFAULT_NORM_FN
    if callable(norm_name_or_fn):
        return norm_name_or_fn
    if isinstance(norm_name_or_fn, str):
        key = norm_name_or_fn.lower()
        if key in NORM_FN_REGISTRY:
            return NORM_FN_REGISTRY[key]
        raise ValueError(
            f"Unknown norm_fn: {norm_name_or_fn}. Available: {list(NORM_FN_REGISTRY.keys())}"
        )
    raise TypeError(f"norm_fn must be a string or callable, got {type(norm_name_or_fn)}")


# Residual Block
class ResBlock(nn.Module):
    node_size: int
    norm_fn: Callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size, dtype=DTYPE)(x0)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        return nn.relu(x + x0)


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    norm_fn: Callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(
            self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
        )(x0)
        x = self.norm_fn(x, training)
        x = nn.relu(x)
        x = nn.Conv(
            self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
        )(x)
        x = self.norm_fn(x, training)
        return nn.relu(x + x0)

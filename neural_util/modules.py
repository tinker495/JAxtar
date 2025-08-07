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
        return BatchNorm(x, training)
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


# Gated Feed-Forward Block using Swish-Gated Linear Unit (SwiGLU)
class SwiGLU(nn.Module):
    output_dim: int
    hidden_dim_factor: int = 2  # ff_dim will be output_dim * factor

    @nn.compact
    def __call__(self, x):
        ff_dim = self.output_dim * self.hidden_dim_factor
        # Project to a larger dimension for gating
        gated_x = nn.Dense(ff_dim * 2, dtype=DTYPE, use_bias=False)(x)

        # Split into two halves for the gate and the value
        x_val, gate = jnp.split(gated_x, 2, axis=-1)

        # Apply Swish activation to the gate and multiply
        return nn.silu(gate) * x_val


ACTIVATION_FN_REGISTRY = {
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "hard_swish": nn.hard_swish,
    "silu": nn.silu,
    "swiglu": SwiGLU,
}


def get_activation_fn(activation_name_or_fn=None):
    if activation_name_or_fn is None:
        return nn.relu
    if callable(activation_name_or_fn):
        return activation_name_or_fn
    if isinstance(activation_name_or_fn, str):
        key = activation_name_or_fn.lower()
        if key in ACTIVATION_FN_REGISTRY:
            return ACTIVATION_FN_REGISTRY[key]
        raise ValueError(
            f"Unknown activation_fn: {activation_name_or_fn}. Available: {list(ACTIVATION_FN_REGISTRY.keys())}"
        )
    raise TypeError(
        f"activation_fn must be a string or callable, got {type(activation_name_or_fn)}"
    )


def get_resblock_fn(resblock_name_or_fn=None):
    if resblock_name_or_fn is None:
        return ResBlock
    if callable(resblock_name_or_fn):
        return resblock_name_or_fn
    if isinstance(resblock_name_or_fn, str):
        key = resblock_name_or_fn.lower()
        if key in RESBLOCK_REGISTRY:
            return RESBLOCK_REGISTRY[key]
        raise ValueError(
            f"Unknown resblock_fn: {resblock_name_or_fn}. Available: {list(RESBLOCK_REGISTRY.keys())}"
        )
    raise TypeError(f"resblock_fn must be a string or callable, got {type(resblock_name_or_fn)}")


# Residual Block
class ResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: Callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        for _ in range(self.hidden_N):
            x = nn.Dense(self.node_size, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
        x = nn.Dense(self.node_size, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        return self.activation(x + x0)


# Pre-activation Residual Block (improved version)
class PreActivationResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: Callable = DEFAULT_NORM_FN
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        residual = x
        # Pre-activation: Norm -> Activation -> Dense
        residual = self.norm_fn(residual, training)
        residual = self.activation(residual)
        for _ in range(self.hidden_N):
            residual = nn.Dense(self.node_size, dtype=DTYPE)(residual)
            residual = self.norm_fn(residual, training)
            residual = self.activation(residual)

        residual = nn.Dense(self.node_size, dtype=DTYPE)(residual)

        # Identity shortcut connection
        return x + residual


# ResBlock type registry for config-driven selection
RESBLOCK_REGISTRY = {
    "standard": ResBlock,
    "preactivation": PreActivationResBlock,
}


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    hidden_N: int = 1
    norm_fn: Callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        for _ in range(self.hidden_N):
            x = nn.Conv(
                self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
            )(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
        x = nn.Conv(
            self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
        )(x)
        x = self.norm_fn(x, training)
        return self.activation(x + x0)

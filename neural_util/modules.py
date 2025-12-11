from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .norm import BatchReNorm as BatchReNorm_
from .norm import DyTan as DyTan_

DTYPE = jnp.bfloat16
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32


def BatchNorm(x, training):
    y = nn.BatchNorm(momentum=0.99, dtype=jnp.float32)(x, use_running_average=not training)
    return y.astype(DTYPE)


def BatchNorm0999(x, training):
    y = nn.BatchNorm(momentum=0.999, dtype=jnp.float32)(x, use_running_average=not training)
    return y.astype(DTYPE)


def BatchReNorm(x, training):
    y = BatchReNorm_(momentum=0.99, dtype=jnp.float32)(x, use_running_average=not training)
    return y.astype(DTYPE)


def BatchReNorm0999(x, training):
    y = BatchReNorm_(momentum=0.999, dtype=jnp.float32)(x, use_running_average=not training)
    return y.astype(DTYPE)


def InstanceNorm(x, training):
    y = nn.InstanceNorm(dtype=jnp.float32)(x)
    return y.astype(DTYPE)


def LayerNorm(x, training):
    y = nn.LayerNorm(dtype=jnp.float32)(x)
    return y.astype(DTYPE)


def GroupNorm(x, training):
    y = nn.GroupNorm(num_groups=10, dtype=jnp.float32)(x)
    return y.astype(DTYPE)


def RMSNorm(x, training):
    y = nn.RMSNorm(dtype=jnp.float32)(x)
    return y.astype(DTYPE)


def DyTan(x, training):
    y = DyTan_(dtype=jnp.float32)(x)
    return y.astype(DTYPE)


DEFAULT_NORM_FN = BatchNorm


# Norm function registry for config-driven selection
NORM_FN_REGISTRY = {
    "batch": BatchNorm,
    "batch0999": BatchNorm0999,
    "batchrenorm": BatchReNorm,
    "batchrenorm0999": BatchReNorm0999,
    "instance": InstanceNorm,
    "layer": LayerNorm,
    "group": GroupNorm,
    "rms": RMSNorm,
    "dytan": DyTan,
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


class Trueswish(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        Beta = self.param("beta", nn.initializers.ones, (x.shape[-1],))
        return x * nn.sigmoid(x * Beta)


class Swiglu(nn.Module):
    hidden_N: int = None
    norm_fn: Callable = None

    @nn.compact
    def __call__(self, x, training=False):
        if self.hidden_N is None:
            hidden_N = x.shape[-1]
        else:
            hidden_N = self.hidden_N
        x = nn.Dense(2 * hidden_N, dtype=DTYPE)(x)
        x, gate = jnp.split(x, 2, axis=-1)
        if self.norm_fn is not None:
            gate = self.norm_fn(gate, training)
        return x * Trueswish()(gate)


ACTIVATION_FN_REGISTRY = {
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "gelu": nn.gelu,
    "swish": lambda x: Trueswish()(x),
    "hard_swish": nn.hard_swish,
    "silu": nn.silu,
    "tanh": nn.tanh,
    "relu6": nn.relu6,
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
    use_swiglu: bool = False

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        out_dim = x0.shape[-1]
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(self.node_size)(x)
            else:
                x = nn.Dense(self.node_size, dtype=DTYPE)(x)
                x = self.norm_fn(x, training)
                x = self.activation(x)
        x = nn.Dense(out_dim, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        return self.activation(x + x0)


# Pre-activation Residual Block (improved version)
class PreActivationResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: Callable = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    zero_init_last: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        residual = x
        out_dim = x.shape[-1]
        # Pre-activation: Norm -> Activation -> Dense
        residual = self.norm_fn(residual, training)
        residual = self.activation(residual)
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                residual = Swiglu(self.node_size)(residual)
            else:
                residual = nn.Dense(self.node_size, dtype=DTYPE)(residual)
                residual = self.activation(residual)

        if self.zero_init_last:
            residual = nn.Dense(out_dim, dtype=DTYPE, kernel_init=nn.initializers.zeros)(residual)
        else:
            residual = nn.Dense(out_dim, dtype=DTYPE)(residual)
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

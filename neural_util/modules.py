from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .norm import BatchReNorm as BatchReNorm_

DTYPE = jnp.bfloat16
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32


def BatchNorm(x, training):
    y = nn.BatchNorm(momentum=0.99, dtype=jnp.float32)(x, use_running_average=not training)
    return y.astype(DTYPE)


def BatchReNorm(x, training):
    y = BatchReNorm_(momentum=0.99, dtype=jnp.float32)(x, use_running_average=not training)
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


DEFAULT_NORM_FN = BatchNorm


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


def swiglu_fn(hidden_N, base_activation=nn.silu, norm_fn=None, param_matching=False):
    target_hidden = hidden_N
    if param_matching:  # match parameter count with non-gated MLP
        target_hidden = max(1, (2 * hidden_N) // 3)

    def _swiglu_fn(x, training=False):
        x = nn.Dense(2 * target_hidden, dtype=DTYPE)(x)
        x, gate = jnp.split(x, 2, axis=-1)
        if norm_fn is not None:
            gate = norm_fn(gate, training)
        return x * base_activation(gate)

    return _swiglu_fn


ACTIVATION_FN_REGISTRY = {
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
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
    activation: Callable = nn.relu
    use_swiglu: bool = False
    residual_scale: float = 0.1
    learnable_residual_scale: bool = True

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        out_dim = x0.shape[-1]
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = swiglu_fn(self.node_size, self.activation, self.norm_fn)(x, training)
            else:
                x = nn.Dense(self.node_size, dtype=DTYPE)(x)
                x = self.norm_fn(x, training)
                x = self.activation(x)
        x = nn.Dense(out_dim, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        if self.learnable_residual_scale:
            residual_scale = self.param(
                "residual_scale",
                lambda key: jnp.asarray(self.residual_scale, dtype=jnp.float32),
            )
        else:
            residual_scale = jnp.asarray(self.residual_scale, dtype=jnp.float32)
        residual_scale = jnp.asarray(residual_scale)
        if residual_scale.dtype != DTYPE:
            residual_scale = residual_scale.astype(DTYPE)
        x = residual_scale * x
        return self.activation(x + x0)


# Pre-activation Residual Block (improved version)
class PreActivationResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: Callable = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    zero_init_last: bool = True
    residual_scale: float = 0.1
    learnable_residual_scale: bool = True

    @nn.compact
    def __call__(self, x, training=False):
        residual = x
        out_dim = x.shape[-1]
        # Pre-activation: Norm -> Activation -> Dense
        residual = self.norm_fn(residual, training)
        residual = self.activation(residual)
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                residual = swiglu_fn(self.node_size, self.activation, self.norm_fn)(
                    residual, training
                )
            else:
                residual = nn.Dense(self.node_size, dtype=DTYPE)(residual)
                residual = self.norm_fn(residual, training)
                residual = self.activation(residual)

        if self.zero_init_last:
            residual = nn.Dense(out_dim, dtype=DTYPE, kernel_init=nn.initializers.zeros)(residual)
        else:
            residual = nn.Dense(out_dim, dtype=DTYPE)(residual)
        # Identity shortcut connection
        if self.learnable_residual_scale:
            residual_scale = self.param(
                "residual_scale",
                lambda key: jnp.asarray(self.residual_scale, dtype=jnp.float32),
            )
        else:
            residual_scale = jnp.asarray(self.residual_scale, dtype=jnp.float32)
        residual_scale = jnp.asarray(residual_scale)
        if residual_scale.dtype != DTYPE:
            residual_scale = residual_scale.astype(DTYPE)
        return x + residual_scale * residual


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
    activation: Callable = nn.relu

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

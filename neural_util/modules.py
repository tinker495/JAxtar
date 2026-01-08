from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp

from .norm import BatchReNorm, DyTan

DTYPE = jnp.float32
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32

DEFAULT_NORM_FN = nn.BatchNorm


# Norm application helper to handle BatchNorm/BatchReNorm running stats.
def apply_norm(norm_fn, x, training):
    if norm_fn is None:
        return x
    fn = norm_fn.func if isinstance(norm_fn, partial) else norm_fn
    if isinstance(fn, type) and issubclass(fn, (nn.BatchNorm, BatchReNorm)):
        return norm_fn()(x, use_running_average=not training)
    return norm_fn()(x)


# Norm function registry for config-driven selection
NORM_FN_REGISTRY = {
    "batch": nn.BatchNorm,
    "batch0999": partial(nn.BatchNorm, momentum=0.999),
    "batchrenorm": BatchReNorm,
    "batchrenorm0999": partial(BatchReNorm, momentum=0.999),
    "instance": nn.InstanceNorm,
    "layer": nn.LayerNorm,
    "group": nn.GroupNorm,
    "rms": nn.RMSNorm,
    "dytan": DyTan,
}


class FunctionalNorm(nn.Module):
    norm_fn: Callable
    dtype: any = DTYPE

    @nn.compact
    def __call__(self, x, training=False):
        return self.norm_fn(x, training, dtype=self.dtype)


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
        Beta = Beta.astype(x.dtype)
        return x * nn.sigmoid(x * Beta)


class Swiglu(nn.Module):
    node_size: int = None
    param_size_equal: bool = True
    norm_fn: nn.Module = None
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        if self.node_size is None:
            node_size = x.shape[-1]
        else:
            node_size = self.node_size
        # Parameter-count matching with a 2-layer MLP (Dense -> Dense) of width H:
        # MLP params ~ d*H + H*d = 2*d*H (bias ignored)
        # SwiGLU params ~ d*(2*h) + h*d = 3*d*h  => h = (2/3)*H
        if self.param_size_equal:
            node_size = max(1, int(round(node_size * 2.0 / 3.0)))
        x = nn.Dense(
            2 * node_size,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(x)
        x, gate = jnp.split(x, 2, axis=-1)
        if self.norm_fn is not None:
            gate = apply_norm(self.norm_fn, gate, training)
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


# Multi-layer Perceptron
class MLP(nn.Module):

    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.hidden_dim, dtype=DTYPE, dot_general_cls=self.dot_general_cls)(x)
        x = apply_norm(self.norm_fn, x, training)
        x = self.activation(x)
        return x


# Pre-activation Multi-layer Perceptron
class preactivation_MLP(nn.Module):
    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    dtype: any = jnp.float32
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x, training=False):
        x = apply_norm(self.norm_fn, x, training)
        x = self.activation(x)
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.01),
            dot_general_cls=self.dot_general_cls,
        )(x)
        return x


# Residual Block
class ResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    use_swiglu: bool = False
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x0, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x0_cast = x0.astype(dtype)
        x = x0_cast
        out_dim = x0.shape[-1]
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)
            else:
                x = MLP(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)
        x = nn.Dense(
            out_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(x)
        x = apply_norm(self.norm_fn, x, training)
        return self.activation(x + x0_cast)


# Pre-activation Residual Block (improved version)
class PreActivationResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    zero_init_last: bool = True
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x0, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x0_cast = x0.astype(dtype)
        out_dim = x0.shape[-1]
        # Pre-activation: Norm -> Activation -> Dense
        x = apply_norm(self.norm_fn, x0_cast, training)
        x = self.activation(x)
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)
            else:
                x = MLP(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)

        if self.zero_init_last:
            x = nn.Dense(
                out_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.zeros,
                dot_general_cls=self.dot_general_cls,
            )(x)
        else:
            x = nn.Dense(
                out_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                dot_general_cls=self.dot_general_cls,
            )(x)
        # Identity shortcut connection
        return x0_cast + x


# Delta Residual Block (Deep Delta Learning)
class DeltaResBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    gate_hidden_dim: int = 128
    eps_k: float = 1e-6
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None

    def _branch_mlp(self, x, out_dim, training):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)
            else:
                x = MLP(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    dot_general_cls=self.dot_general_cls,
                )(x, training)
        x = nn.Dense(
            out_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(x)
        return x

    @nn.compact
    def __call__(self, x0, training=False):
        dtype = self.dtype
        x = x0.astype(dtype)
        if x.ndim == 2:
            pool_d = x
            pool_v = jnp.mean(x, axis=-1, keepdims=True)
            dv = 1
        elif x.ndim == 3:
            pool_d = jnp.mean(x, axis=-1)
            pool_v = jnp.mean(x, axis=-2)
            dv = x.shape[-1]
        else:
            raise ValueError(f"DeltaResBlock expects 2D or 3D input, got shape {x.shape}")

        k = self._branch_mlp(pool_d, x.shape[-2] if x.ndim == 3 else x.shape[-1], training)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + self.eps_k)

        v = self._branch_mlp(pool_v, dv, training)

        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        beta_hidden = nn.Dense(
            self.gate_hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(pool_d)
        beta_hidden = jnp.tanh(beta_hidden)
        beta = nn.Dense(
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(beta_hidden)
        beta = 2.0 * nn.sigmoid(beta).squeeze(-1)

        if x.ndim == 2:
            proj = jnp.sum(k * x, axis=-1, keepdims=True)
            update = k * (v - proj)
            x = x + beta[:, None] * update
        else:
            proj = jnp.einsum("bd,bdv->bv", k, x)
            update = k[:, :, None] * (v - proj)[:, None, :]
            x = x + beta[:, None, None] * update

        return self.activation(x)


# ResBlock type registry for config-driven selection
RESBLOCK_REGISTRY = {
    "standard": ResBlock,
    "preactivation": PreActivationResBlock,
    "delta": DeltaResBlock,
}


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        for _ in range(self.hidden_N):
            x = nn.Conv(
                self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
            )(x)
            x = apply_norm(self.norm_fn, x, training)
            x = self.activation(x)
        x = nn.Conv(
            self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
        )(x)
        x = apply_norm(self.norm_fn, x, training)
        return self.activation(x + x0)

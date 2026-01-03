from functools import partial
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .norm import BatchReNorm, DyTan

DTYPE = jnp.bfloat16
# Use float32 for numerically sensitive heads / losses.
HEAD_DTYPE = jnp.float32

DEFAULT_NORM_FN = nn.BatchNorm


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
        x = nn.Dense(2 * node_size, dtype=dtype, param_dtype=param_dtype)(x)
        x, gate = jnp.split(x, 2, axis=-1)
        if self.norm_fn is not None:
            gate = self.norm_fn()(gate, training)
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

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
        x = self.norm_fn()(x, training)
        x = self.activation(x)
        return x


# Pre-activation Multi-layer Perceptron
class preactivation_MLP(nn.Module):
    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x, training=False):
        x = self.norm_fn()(x, training)
        x = self.activation(x)
        x = nn.Dense(
            self.hidden_dim, dtype=self.dtype, kernel_init=nn.initializers.normal(stddev=0.01)
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
                )(x, training)
            else:
                x = MLP(self.node_size, norm_fn=self.norm_fn, activation=self.activation)(
                    x, training
                )
        x = nn.Dense(out_dim, dtype=dtype, param_dtype=param_dtype)(x)
        x = self.norm_fn()(x, training)
        return self.activation(x + x0_cast)


def sinkhorn_knopp(logits, num_iters=20, eps=1e-6):
    weights = jnp.exp(logits)
    for _ in range(num_iters):
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + eps)
        weights = weights / (jnp.sum(weights, axis=-2, keepdims=True) + eps)
    return weights


class MHCHyperConnections(nn.Module):
    n_streams: int = 4
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01
    rms_norm_epsilon: float = 1e-6
    dtype: any = DTYPE
    param_dtype: any = None

    @nn.compact
    def __call__(self, x_stream, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        n_streams = self.n_streams
        x_cast = x_stream.astype(dtype)
        x_norm = nn.RMSNorm(epsilon=self.rms_norm_epsilon, dtype=dtype)(x_cast)
        feature_dim = x_stream.shape[-1]

        theta_pre = self.param(
            "theta_pre",
            nn.initializers.normal(stddev=0.02),
            (feature_dim,),
            param_dtype,
        )
        theta_post = self.param(
            "theta_post",
            nn.initializers.normal(stddev=0.02),
            (feature_dim,),
            param_dtype,
        )
        theta_res = self.param(
            "theta_res",
            nn.initializers.normal(stddev=0.02),
            (n_streams, feature_dim),
            param_dtype,
        )
        b_pre = self.param("b_pre", nn.initializers.zeros, (n_streams,), param_dtype)
        b_post = self.param("b_post", nn.initializers.zeros, (n_streams,), param_dtype)
        b_res = self.param("b_res", nn.initializers.zeros, (n_streams, n_streams), param_dtype)
        alpha_pre = self.param(
            "alpha_pre", nn.initializers.constant(self.alpha_init), (1,), param_dtype
        )
        alpha_post = self.param(
            "alpha_post", nn.initializers.constant(self.alpha_init), (1,), param_dtype
        )
        alpha_res = self.param(
            "alpha_res", nn.initializers.constant(self.alpha_init), (1,), param_dtype
        )

        h_pre_dyn = jnp.einsum("c,bnc->bn", theta_pre, x_norm)
        h_post_dyn = jnp.einsum("c,bnc->bn", theta_post, x_norm)
        h_res_dyn = jnp.einsum("ic,bjc->bij", theta_res, x_norm)

        r = jnp.linalg.norm(x_cast, axis=(-2, -1), keepdims=True) / jnp.sqrt(
            n_streams * feature_dim
        )
        inv_r = 1.0 / (r[..., 0, 0] + 1e-6)

        h_pre_logits = alpha_pre * jnp.tanh(h_pre_dyn) * inv_r[:, None] + b_pre
        h_post_logits = alpha_post * jnp.tanh(h_post_dyn) * inv_r[:, None] + b_post
        h_res_logits = alpha_res * jnp.tanh(h_res_dyn) * inv_r[:, None, None] + b_res

        h_pre = nn.sigmoid(h_pre_logits)
        h_post = 2.0 * nn.sigmoid(h_post_logits)
        h_res = sinkhorn_knopp(h_res_logits, num_iters=self.sinkhorn_iters)
        return h_pre, h_post, h_res


class MHCMLPBlock(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    dtype: any = DTYPE
    param_dtype: any = None

    @nn.compact
    def __call__(self, x, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x = x.astype(dtype)
        out_dim = x.shape[-1]
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )(x, training)
            else:
                x = MLP(self.node_size, norm_fn=self.norm_fn, activation=self.activation)(
                    x, training
                )
        x = nn.Dense(out_dim, dtype=dtype, param_dtype=param_dtype)(x)
        x = self.norm_fn()(x, training)
        return self.activation(x)


class MHCLayer(nn.Module):
    node_size: int
    hidden_N: int = 1
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: Callable = nn.relu
    use_swiglu: bool = False
    n_streams: int = 4
    sinkhorn_iters: int = 20
    alpha_init: float = 0.01
    rms_norm_epsilon: float = 1e-6
    dtype: any = DTYPE
    param_dtype: any = None

    @nn.compact
    def __call__(self, x_stream, training=False):
        h_pre, h_post, h_res = MHCHyperConnections(
            n_streams=self.n_streams,
            sinkhorn_iters=self.sinkhorn_iters,
            alpha_init=self.alpha_init,
            rms_norm_epsilon=self.rms_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x_stream, training)
        x_pre = jnp.einsum("bn,bnc->bc", h_pre, x_stream)
        x_post = MHCMLPBlock(
            self.node_size,
            hidden_N=self.hidden_N,
            norm_fn=self.norm_fn,
            activation=self.activation,
            use_swiglu=self.use_swiglu,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x_pre, training)
        x_res = jnp.einsum("bij,bjc->bic", h_res, x_stream)
        x_post_stream = jnp.einsum("bn,bc->bnc", h_post, x_post)
        return x_res + x_post_stream


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

    @nn.compact
    def __call__(self, x0, training=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x0_cast = x0.astype(dtype)
        out_dim = x0.shape[-1]
        # Pre-activation: Norm -> Activation -> Dense
        x = self.norm_fn()(x0_cast, training)
        x = self.activation(x)
        for _ in range(self.hidden_N):
            if self.use_swiglu:
                x = Swiglu(
                    self.node_size,
                    norm_fn=self.norm_fn,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )(x, training)
            else:
                x = MLP(self.node_size, norm_fn=self.norm_fn, activation=self.activation)(
                    x, training
                )

        if self.zero_init_last:
            x = nn.Dense(
                out_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.zeros,
            )(x)
        else:
            x = nn.Dense(out_dim, dtype=dtype, param_dtype=param_dtype)(x)
        # Identity shortcut connection
        return x0_cast + x


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
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x0, training=False):
        x = x0
        for _ in range(self.hidden_N):
            x = nn.Conv(
                self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
            )(x)
            x = self.norm_fn()(x, training)
            x = self.activation(x)
        x = nn.Conv(
            self.filters, self.kernel_size, strides=self.strides, padding="SAME", dtype=DTYPE
        )(x)
        x = self.norm_fn()(x, training)
        return self.activation(x + x0)

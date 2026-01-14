import math
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from .norm import BatchReNorm, DyTan

DTYPE = jnp.bfloat16
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


# Shared helpers for MoE blocks.
def _moe_dense_gates(logits, router_temperature):
    temp = jnp.maximum(router_temperature, 1e-6)
    scaled_logits = logits.astype(jnp.float32) / temp
    return jax.nn.softmax(scaled_logits, axis=-1)


def _moe_topk_gates(logits, num_experts, top_k, router_temperature):
    temp = jnp.maximum(router_temperature, 1e-6)
    scaled_logits = logits.astype(jnp.float32) / temp
    top_k = min(int(top_k), num_experts)
    top_vals, top_idx = jax.lax.top_k(scaled_logits, top_k)
    top_gates = jax.nn.softmax(top_vals, axis=-1)

    # Use dense probabilities for the auxiliary loss to ensure gradients flow to all experts.
    # This prevents the "dead expert" problem where unselected experts never get updated.
    gate_probs = jax.nn.softmax(scaled_logits, axis=-1)
    return top_idx, top_gates, gate_probs


class MoEExpert(nn.Module):
    node_size: int
    out_dim: int
    hidden_N: int = 1
    norm_fn: nn.Module = nn.LayerNorm
    activation: Callable = nn.relu
    use_swiglu: bool = False
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None
    zero_init_last: bool = False
    apply_final_norm: bool = True

    @nn.compact
    def __call__(self, x, training=False):
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

        dense_kwargs = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )
        if self.zero_init_last:
            dense_kwargs["kernel_init"] = nn.initializers.zeros
        x = nn.Dense(self.out_dim, **dense_kwargs)(x)
        if self.apply_final_norm:
            x = apply_norm(self.norm_fn, x, training)
        return x


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


# MoE Residual Block
class MoEResBlock(nn.Module):
    node_size: int
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.0
    min_capacity: int = 1
    hidden_N: int = 1
    norm_fn: nn.Module = nn.LayerNorm
    activation: Callable = nn.relu
    use_swiglu: bool = False
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None
    router_temperature: float = 1.0
    router_noise_std: float = 0.0

    @nn.compact
    def __call__(self, x0, training=False, capture_aux=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x0_cast = x0.astype(dtype)

        # Handle 3D input (Batch, Seq, Dim) by flattening to (Batch*Seq, Dim)
        input_ndim = x0_cast.ndim
        if input_ndim == 3:
            B, S, D = x0_cast.shape
            x_in = x0_cast.reshape(B * S, D)
        else:
            x_in = x0_cast

        out_dim = x_in.shape[-1]

        logits = nn.Dense(
            self.num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(x_in)
        if training and self.router_noise_std > 0.0:
            noise = jax.random.normal(self.make_rng("params"), logits.shape, dtype=logits.dtype)
            logits = logits + noise * self.router_noise_std

        use_sparse = self.top_k is not None and self.top_k > 0 and self.top_k < self.num_experts
        if use_sparse:
            top_idx, top_gates, gate_probs = _moe_topk_gates(
                logits, self.num_experts, self.top_k, self.router_temperature
            )
            top_k = top_idx.shape[1]
            batch_size = x_in.shape[0]
            capacity = max(
                self.min_capacity,
                int(math.ceil(self.capacity_factor * batch_size * top_k / self.num_experts)),
            )

            if top_k == 1:
                x_flat = x_in
            else:
                x_flat = jnp.repeat(x_in, top_k, axis=0)

            expert_idx = top_idx.reshape(-1)
            gate_flat = top_gates.reshape(-1).astype(dtype)
            expert_one_hot = jax.nn.one_hot(expert_idx, self.num_experts, dtype=jnp.int32)
            positions = jnp.cumsum(expert_one_hot, axis=0) - 1
            pos = jnp.sum(positions * expert_one_hot, axis=1).astype(jnp.int32)
            mask = (pos < capacity).astype(dtype)
            pos_one_hot = jax.nn.one_hot(pos, capacity, dtype=dtype)

            dispatch = expert_one_hot.astype(dtype)[:, :, None] * pos_one_hot[:, None, :]
            dispatch = dispatch * mask[:, None, None]
            expert_inputs = jnp.einsum("bec,bd->ecd", dispatch, x_flat)

            # Use nn.vmap to execute all experts in parallel (batched execution)
            # This is much more efficient than a list comprehension on GPU.
            experts = nn.vmap(
                MoEExpert,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=0,
                out_axes=0,
                axis_size=self.num_experts,
            )(
                self.node_size,
                out_dim,
                hidden_N=self.hidden_N,
                norm_fn=self.norm_fn,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=dtype,
                param_dtype=param_dtype,
                dot_general_cls=self.dot_general_cls,
                apply_final_norm=True,
                name="experts",
            )(
                expert_inputs, training
            )

            combine = dispatch * gate_flat[:, None, None]
            y_flat = jnp.einsum("bec,ecd->bd", combine, experts)
            mixture = y_flat.reshape(batch_size, top_k, out_dim).sum(axis=1)
            out = self.activation(mixture + x_in)
        else:
            gate_probs = _moe_dense_gates(logits, self.router_temperature)
            gate_probs_mix = gate_probs.astype(dtype)

            experts = nn.vmap(
                MoEExpert,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=(None, None),
                out_axes=0,
                axis_size=self.num_experts,
            )(
                self.node_size,
                out_dim,
                hidden_N=self.hidden_N,
                norm_fn=self.norm_fn,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=dtype,
                param_dtype=param_dtype,
                dot_general_cls=self.dot_general_cls,
                apply_final_norm=True,
                name="experts",
            )(
                x_in, training
            )

            # experts: (E, B, D), gate_probs: (B, E)
            # einsum: "ebd,be->bd"
            mixture = jnp.einsum("ebd,be->bd", experts, gate_probs_mix)
            out = self.activation(mixture + x_in)

        if input_ndim == 3:
            out = out.reshape(B, S, D)

        if not capture_aux:
            return out

        gate_mean = jnp.mean(gate_probs, axis=0)

        # Standard load balancing loss: N * sum(importance * load)
        # importance = gate_mean
        # load = fraction of tokens assigned to each expert
        expert_idx = top_idx.reshape(-1)
        expert_one_hot = jax.nn.one_hot(expert_idx, self.num_experts, dtype=jnp.float32)
        load = jnp.mean(expert_one_hot, axis=0)

        balance_loss = self.num_experts * jnp.sum(gate_mean * load)

        # Router z-loss for stability: mean(log(sum(exp(x)))^2)
        log_sum_exp = jax.scipy.special.logsumexp(logits, axis=-1)
        router_z_loss = jnp.mean(jnp.square(log_sum_exp))

        gate_entropy = -jnp.sum(gate_mean * jnp.log(gate_mean + 1e-8))
        return out, {
            "balance_loss": balance_loss,
            "gate_entropy": gate_entropy,
            "router_z_loss": router_z_loss,
        }


# MoE Pre-activation Residual Block
class MoEPreActivationResBlock(nn.Module):
    node_size: int
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.0
    min_capacity: int = 1
    hidden_N: int = 1
    norm_fn: nn.Module = nn.LayerNorm
    activation: Callable = nn.relu
    use_swiglu: bool = False
    zero_init_last: bool = True
    dtype: any = DTYPE
    param_dtype: any = None
    dot_general_cls: Any = None
    router_temperature: float = 1.0
    router_noise_std: float = 0.0

    @nn.compact
    def __call__(self, x0, training=False, capture_aux=False):
        dtype = self.dtype
        param_dtype = self.param_dtype if self.param_dtype is not None else HEAD_DTYPE
        x0_cast = x0.astype(dtype)

        # Handle 3D input
        input_ndim = x0_cast.ndim
        if input_ndim == 3:
            B, S, D = x0_cast.shape
            x0_flat = x0_cast.reshape(B * S, D)
        else:
            x0_flat = x0_cast

        # Norm and activation on flattened or original input
        # Note: applying norm on (B*S, D) is often same as (B, S, D) for LayerNorm/RMSNorm but different for BatchNorm
        # BatchNorm usually expects (B, S, D).
        # But here we are making MoE experts work on tokens.
        # If we use BatchNorm, we should probably apply it before flattening?
        # PreActivation: Norm -> act -> Dense.
        # Let's apply norm on valid shape first.

        x = apply_norm(self.norm_fn, x0_flat, training)
        x = self.activation(x)

        # Now flatten for MoE
        if input_ndim == 3:
            x_in = x.reshape(B * S, D)
        else:
            x_in = x

        out_dim = x_in.shape[-1]

        logits = nn.Dense(
            self.num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            dot_general_cls=self.dot_general_cls,
        )(x_in)
        if training and self.router_noise_std > 0.0:
            noise = jax.random.normal(self.make_rng("params"), logits.shape, dtype=logits.dtype)
            logits = logits + noise * self.router_noise_std

        use_sparse = self.top_k is not None and self.top_k > 0 and self.top_k < self.num_experts
        if use_sparse:
            top_idx, top_gates, gate_probs = _moe_topk_gates(
                logits, self.num_experts, self.top_k, self.router_temperature
            )
            top_k = top_idx.shape[1]
            batch_size = x_in.shape[0]
            capacity = max(
                self.min_capacity,
                int(math.ceil(self.capacity_factor * batch_size * top_k / self.num_experts)),
            )

            if top_k == 1:
                x_flat = x_in
            else:
                x_flat = jnp.repeat(x_in, top_k, axis=0)

            expert_idx = top_idx.reshape(-1)
            gate_flat = top_gates.reshape(-1).astype(dtype)
            expert_one_hot = jax.nn.one_hot(expert_idx, self.num_experts, dtype=jnp.int32)
            positions = jnp.cumsum(expert_one_hot, axis=0) - 1
            pos = jnp.sum(positions * expert_one_hot, axis=1).astype(jnp.int32)
            mask = (pos < capacity).astype(dtype)
            pos_one_hot = jax.nn.one_hot(pos, capacity, dtype=dtype)

            dispatch = expert_one_hot.astype(dtype)[:, :, None] * pos_one_hot[:, None, :]
            dispatch = dispatch * mask[:, None, None]
            expert_inputs = jnp.einsum("bec,bd->ecd", dispatch, x_flat)

            # Use nn.vmap to execute all experts in parallel
            experts = nn.vmap(
                MoEExpert,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=0,
                out_axes=0,
                axis_size=self.num_experts,
            )(
                self.node_size,
                out_dim,
                hidden_N=self.hidden_N,
                norm_fn=self.norm_fn,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=dtype,
                param_dtype=param_dtype,
                dot_general_cls=self.dot_general_cls,
                zero_init_last=self.zero_init_last,
                apply_final_norm=False,
                name="experts",
            )(
                expert_inputs, training
            )

            combine = dispatch * gate_flat[:, None, None]
            y_flat = jnp.einsum("bec,ecd->bd", combine, experts)
            mixture = y_flat.reshape(batch_size, top_k, out_dim).sum(axis=1)

            if input_ndim == 3:
                mixture = mixture.reshape(B, S, D)

            out = x0_cast + mixture
        else:
            gate_probs = _moe_dense_gates(logits, self.router_temperature)
            gate_probs_mix = gate_probs.astype(dtype)

            experts = nn.vmap(
                MoEExpert,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=(None, None),
                out_axes=0,
                axis_size=self.num_experts,
            )(
                self.node_size,
                out_dim,
                hidden_N=self.hidden_N,
                norm_fn=self.norm_fn,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dtype=dtype,
                param_dtype=param_dtype,
                dot_general_cls=self.dot_general_cls,
                zero_init_last=self.zero_init_last,
                apply_final_norm=False,
                name="experts",
            )(
                x_in, training
            )

            # experts: (E, B, D), gate_probs: (B, E)
            # einsum: "ebd,be->bd"
            mixture = jnp.einsum("ebd,be->bd", experts, gate_probs_mix)

            if input_ndim == 3:
                mixture = mixture.reshape(B, S, D)

            out = x0_cast + mixture

        if not capture_aux:
            return out

        gate_mean = jnp.mean(gate_probs, axis=0)

        # Standard load balancing loss: N * sum(importance * load)
        # importance = gate_mean
        # load = fraction of tokens assigned to each expert
        expert_idx = top_idx.reshape(-1)
        expert_one_hot = jax.nn.one_hot(expert_idx, self.num_experts, dtype=jnp.float32)
        load = jnp.mean(expert_one_hot, axis=0)

        balance_loss = self.num_experts * jnp.sum(gate_mean * load)

        # Router z-loss for stability: mean(log(sum(exp(x)))^2)
        log_sum_exp = jax.scipy.special.logsumexp(logits, axis=-1)
        router_z_loss = jnp.mean(jnp.square(log_sum_exp))

        gate_entropy = -jnp.sum(gate_mean * jnp.log(gate_mean + 1e-8))
        return out, {
            "balance_loss": balance_loss,
            "gate_entropy": gate_entropy,
            "router_z_loss": router_z_loss,
        }


# ResBlock type registry for config-driven selection
RESBLOCK_REGISTRY = {
    "standard": ResBlock,
    "preactivation": PreActivationResBlock,
    "delta": DeltaResBlock,
    "moe": MoEResBlock,
    "moe_preactivation": MoEPreActivationResBlock,
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

"""Puzzle-agnostic BatchNorm-free Transformer for discrete-variable distance models."""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn

from neural_util.aqt_utils import build_aqt_dot_general
from neural_util.basemodel.base import DistanceModel
from neural_util.dtypes import DTYPE, HEAD_DTYPE, PARAM_DTYPE


def _attention_implementation(head_dim: int) -> Literal["cudnn", "xla"]:
    """Select cuDNN FlashAttention on GPU when supported, else fall back to XLA."""
    if (
        jax.default_backend() == "gpu"
        and DTYPE in (jnp.bfloat16, jnp.float16)
        and head_dim <= 128
        and head_dim % 8 == 0
    ):
        return "cudnn"
    return "xla"


class DiscreteVarTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with QK-RMSNorm FlashAttention and SwiGLU FFN."""

    d_model: int
    num_heads: int
    d_ff: int
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        head_dim = self.d_model // self.num_heads
        if head_dim * self.num_heads != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})."
            )

        h = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="attn_norm")(x)
        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="qkv",
        )(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        bsz, seq_len, _ = q.shape
        q = q.reshape((bsz, seq_len, self.num_heads, head_dim))
        k = k.reshape((bsz, seq_len, self.num_heads, head_dim))
        v = v.reshape((bsz, seq_len, self.num_heads, head_dim))

        # QK-RMSNorm stabilizes attention logits across learning rates without BatchNorm.
        q = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="q_norm")(q)
        k = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="k_norm")(k)
        attn = jax.nn.dot_product_attention(
            q, k, v, implementation=_attention_implementation(head_dim)
        ).reshape((bsz, seq_len, self.d_model))
        attn_out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="out_proj",
        )(attn)
        x = (x + attn_out).astype(DTYPE)

        h = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="ffn_norm")(x)
        gate_up = nn.Dense(
            2 * self.d_ff,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="ffn_gate_up",
        )(h)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        ffn_hidden = nn.silu(gate) * up
        ffn_out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="ffn_down",
        )(ffn_hidden)
        return (x + ffn_out).astype(DTYPE)


class _LatentCrossAttention(nn.Module):
    """Compress T discrete-variable embeddings into M learned latent tokens via cross-attention."""

    num_latents: int
    d_model: int
    num_heads: int
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        head_dim = self.d_model // self.num_heads
        if head_dim * self.num_heads != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})."
            )
        bsz, seq_len, _ = tokens.shape
        latents = self.param(
            "latents",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_latents, self.d_model),
            PARAM_DTYPE,
        ).astype(DTYPE)
        latents = jnp.broadcast_to(latents, (bsz, self.num_latents, self.d_model))

        q_in = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="q_in_norm")(latents)
        kv_in = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="kv_in_norm")(tokens)
        q = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="q_proj",
        )(q_in).reshape((bsz, self.num_latents, self.num_heads, head_dim))
        kv = nn.Dense(
            2 * self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="kv_proj",
        )(kv_in)
        k, v = jnp.split(kv, 2, axis=-1)
        k = k.reshape((bsz, seq_len, self.num_heads, head_dim))
        v = v.reshape((bsz, seq_len, self.num_heads, head_dim))

        q = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="q_norm")(q)
        k = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="k_norm")(k)
        attn = jax.nn.dot_product_attention(
            q, k, v, implementation=_attention_implementation(head_dim)
        ).reshape((bsz, self.num_latents, self.d_model))
        out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="out_proj",
        )(attn)
        return (latents + out).astype(DTYPE)


class _RMSNormSwiGLUResBlock(nn.Module):
    """Row-independent Pre-RMSNorm SwiGLU residual block for single-vector hybrid trunks."""

    dim: int
    ff_dim: int
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="norm")(x)
        gate_up = nn.Dense(
            2 * self.ff_dim,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="gate_up",
        )(h)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        hidden = nn.silu(gate) * up
        out = nn.Dense(
            self.dim,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            dot_general_cls=self.dot_general_cls,
            name="down",
        )(hidden)
        return (x + out).astype(DTYPE)


class _DiscreteVarTransformerCore(nn.Module):
    action_size: int
    vocab_size: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    slot_proj_dim: int
    num_latents: int = 0
    mlp_layers: int = 0
    mlp_dim: int = 512
    mlp_ff_dim: int = 576
    use_remat: bool = True
    dot_general_cls: Any = None

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        bsz, num_tokens = x.shape
        slot_offsets = jnp.arange(num_tokens, dtype=jnp.int32)[None, :] * self.vocab_size
        pair_ids = slot_offsets + x.astype(jnp.int32)
        h = nn.Embed(
            num_embeddings=num_tokens * self.vocab_size,
            features=self.d_model,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="pair_embed",
        )(pair_ids)

        remat_active = training and self.use_remat
        if self.num_latents > 0:
            cross_cls = nn.remat(_LatentCrossAttention) if remat_active else _LatentCrossAttention
            h = cross_cls(
                num_latents=self.num_latents,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dot_general_cls=self.dot_general_cls,
                name="latent_cross_attn",
            )(h)
            out_tokens = self.num_latents
        else:
            out_tokens = num_tokens

        block_cls = (
            nn.remat(DiscreteVarTransformerBlock) if remat_active else DiscreteVarTransformerBlock
        )
        for i in range(self.num_layers):
            h = block_cls(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dot_general_cls=self.dot_general_cls,
                name=f"block_{i}",
            )(h)

        # Preserve per-slot final features before projecting to single-vector trunk or head.
        h = nn.RMSNorm(dtype=DTYPE, param_dtype=PARAM_DTYPE, name="slot_norm")(h)
        slot_feats = nn.Dense(
            self.slot_proj_dim,
            use_bias=False,
            dtype=DTYPE,
            param_dtype=PARAM_DTYPE,
            name="slot_proj",
        )(h)
        pooled = slot_feats.reshape((bsz, out_tokens * self.slot_proj_dim))

        if self.mlp_layers > 0:
            if pooled.shape[-1] != self.mlp_dim:
                pooled = nn.Dense(
                    self.mlp_dim,
                    use_bias=False,
                    dtype=DTYPE,
                    param_dtype=PARAM_DTYPE,
                    dot_general_cls=self.dot_general_cls,
                    name="mlp_in_proj",
                )(pooled)
            for i in range(self.mlp_layers):
                pooled = _RMSNormSwiGLUResBlock(
                    dim=self.mlp_dim,
                    ff_dim=self.mlp_ff_dim,
                    dot_general_cls=self.dot_general_cls,
                    name=f"mlp_block_{i}",
                )(pooled)

        pooled = nn.RMSNorm(dtype=HEAD_DTYPE, param_dtype=PARAM_DTYPE, name="head_norm")(pooled)
        return nn.Dense(
            self.action_size,
            dtype=HEAD_DTYPE,
            param_dtype=PARAM_DTYPE,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="head",
        )(pooled)


class DiscreteVarTransformerModel(DistanceModel):
    """Row-independent Transformer over fixed-length integer token states `(B, T)`.

    Each input position `i` with discrete value `v` indexes a learned `(i, v)` pair
    embedding without puzzle-specific preprocessing or BatchNorm statistics. Setting
    `mlp_layers > 0` creates a hybrid architecture: shallow token attention followed by
    a fast single-vector Pre-RMSNorm SwiGLU residual trunk.
    """

    vocab_size: int = 6
    num_layers: int = 6
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512
    slot_proj_dim: int = 8
    num_latents: int = 0
    mlp_layers: int = 0
    mlp_dim: int = 512
    mlp_ff_dim: int = 576
    use_remat: bool = True
    eval_chunk_size: int = 16384
    aqt_cfg: Any = None
    quant_mode: Any = None

    @property
    def supports_row_independent_inference(self) -> bool:
        return type(self) is DiscreteVarTransformerModel and self.aqt_cfg is None

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        if x.ndim != 2:
            raise ValueError(
                f"DiscreteVarTransformerModel expects 2D integer tokens (B, T), got {x.shape}."
            )
        if self.vocab_size <= 0 or self.num_layers <= 0 or self.slot_proj_dim <= 0:
            raise ValueError("vocab_size, num_layers, and slot_proj_dim must be positive integers.")
        if self.num_latents < 0 or self.mlp_layers < 0:
            raise ValueError("num_latents and mlp_layers must be non-negative.")

        aqt_dg = None
        if self.aqt_cfg is not None:
            mode = self.quant_mode if self.quant_mode is not None else aqt_flax.QuantMode.TRAIN
            aqt_dg = build_aqt_dot_general(self.aqt_cfg, mode)

        core = _DiscreteVarTransformerCore(
            action_size=self.action_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            slot_proj_dim=self.slot_proj_dim,
            num_latents=self.num_latents,
            mlp_layers=self.mlp_layers,
            mlp_dim=self.mlp_dim,
            mlp_ff_dim=self.mlp_ff_dim,
            use_remat=self.use_remat,
            dot_general_cls=aqt_dg,
            name="core",
        )

        bsz = x.shape[0]
        # Chunk large evaluation batches (e.g. DAVI 131K dataset minibatches) so token-expanded
        # activations `(B, T, d)` stay bounded in GPU memory without altering predictions.
        if not training and self.eval_chunk_size > 0 and bsz > self.eval_chunk_size:
            chunk = self.eval_chunk_size
            n_full = bsz // chunk
            rem = bsz % chunk
            chunks = x[: n_full * chunk].reshape((n_full, chunk, x.shape[1]))
            first = core(chunks[0], training=False)
            parts = [first]
            if n_full > 1:
                _, rest = jax.lax.scan(
                    lambda _, c: (None, core(c, training=False)),
                    None,
                    chunks[1:],
                )
                parts.append(rest.reshape(((n_full - 1) * chunk, self.action_size)))
            if rem > 0:
                parts.append(core(x[n_full * chunk :], training=False))
            return jnp.concatenate(parts, axis=0)

        return core(x, training=training)


class DiscreteVarTransformerHybridModel(DiscreteVarTransformerModel):
    """4M hybrid preset: 2-layer FlashAttention frontend + 4-layer Pre-RMSNorm SwiGLU trunk."""

    num_layers: int = 2
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 128
    slot_proj_dim: int = 16
    num_latents: int = 0
    mlp_layers: int = 4
    mlp_dim: int = 512
    mlp_ff_dim: int = 576
    use_remat: bool = False

    @property
    def supports_row_independent_inference(self) -> bool:
        return type(self) is DiscreteVarTransformerHybridModel and self.aqt_cfg is None

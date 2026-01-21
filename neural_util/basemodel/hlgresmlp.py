from typing import Any

import jax
import jax.numpy as jnp
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn

from neural_util.aqt_utils import build_aqt_dot_general
from neural_util.basemodel.base import DistanceHLGModel
from neural_util.basemodel.selfpredictive import SelfPredictiveDistanceHLGModel
from neural_util.dtypes import DTYPE, HEAD_DTYPE, PARAM_DTYPE
from neural_util.modules import (
    DEFAULT_NORM_FN,
    MLP,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
    get_activation_fn,
    preactivation_MLP,
)


class HLGResMLPModel(DistanceHLGModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0
    aqt_cfg: Any = None
    quant_mode: Any = None

    def setup(self):
        super().setup()
        # Resolve activation to callable if it's a string
        resolved_activation = get_activation_fn(self.activation)

        aqt_dg = None
        if self.aqt_cfg is not None:
            mode = self.quant_mode if self.quant_mode is not None else aqt_flax.QuantMode.TRAIN
            aqt_dg = build_aqt_dot_general(self.aqt_cfg, mode)

        self.initial_mlp = (
            Swiglu(
                self.initial_dim,
                norm_fn=self.norm_fn,
                dtype=DTYPE,
                param_dtype=PARAM_DTYPE,
                dot_general_cls=aqt_dg,
            )
            if self.use_swiglu
            else MLP(
                self.initial_dim,
                norm_fn=self.norm_fn,
                activation=resolved_activation,
                dot_general_cls=aqt_dg,
            )
        )
        self.second_mlp = (
            (
                Swiglu(
                    self.hidden_dim,
                    norm_fn=self.norm_fn,
                    dtype=DTYPE,
                    param_dtype=PARAM_DTYPE,
                    dot_general_cls=aqt_dg,
                )
                if self.use_swiglu
                else MLP(
                    self.hidden_dim,
                    norm_fn=self.norm_fn,
                    activation=resolved_activation,
                    dot_general_cls=aqt_dg,
                )
            )
            if self.resblock_fn != PreActivationResBlock
            else nn.Dense(
                self.hidden_dim, dtype=DTYPE, param_dtype=PARAM_DTYPE, dot_general_cls=aqt_dg
            )
        )

        self.resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=resolved_activation,
                use_swiglu=self.use_swiglu,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.Res_N - self.tail_head_precision)
        ]
        self.tail_head_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=resolved_activation,
                use_swiglu=self.use_swiglu,
                dtype=HEAD_DTYPE,
                param_dtype=HEAD_DTYPE,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.tail_head_precision)
        ]
        self.final_dense = (
            preactivation_MLP(
                self.action_size * self.categorial_n, dtype=HEAD_DTYPE, dot_general_cls=aqt_dg
            )
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size * self.categorial_n,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
                dot_general_cls=aqt_dg,
            )
        )

    def get_logits(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        for resblock in self.resblocks:
            x = resblock(x, training)
        for resblock in self.tail_head_resblocks:
            x = resblock(x, training)

        if isinstance(self.final_dense, nn.Dense):
            x = x.astype(HEAD_DTYPE)
            x = self.final_dense(x)
        else:
            x = self.final_dense(x, training)

        x = x.reshape(x.shape[:-1] + (self.action_size, self.categorial_n))
        return x


class SelfPredictiveHLGResMLPModel(SelfPredictiveDistanceHLGModel):

    embedding_Res_N: int = 3
    distances_Res_N: int = 1
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    projection_dim: int = 128
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0
    path_action_size: int = 12

    def setup(self):
        super().setup()
        # Resolve activation to callable if it's a string
        resolved_activation = get_activation_fn(self.activation)

        # Initial and second MLP (following SelfPredictiveResMLPModel pattern)
        self.initial_mlp = (
            Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE)
            if self.use_swiglu
            else MLP(self.initial_dim, norm_fn=self.norm_fn, activation=resolved_activation)
        )
        self.second_mlp = (
            (
                Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE)
                if self.use_swiglu
                else MLP(self.hidden_dim, norm_fn=self.norm_fn, activation=resolved_activation)
            )
            if self.resblock_fn != PreActivationResBlock
            else nn.Dense(self.hidden_dim, dtype=DTYPE)
        )

        # Embedding resblocks (states -> latents)
        self.embedding_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=resolved_activation,
                use_swiglu=self.use_swiglu,
            )
            for _ in range(self.embedding_Res_N)
        ]

        # Distances resblocks (latents -> logits)
        self.distances_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=resolved_activation,
                use_swiglu=self.use_swiglu,
            )
            for _ in range(self.distances_Res_N)
        ]

        # Final dense for HLG (outputs action_size * categorial_n)
        self.final_dense = (
            preactivation_MLP(self.action_size * self.categorial_n, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size * self.categorial_n,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
            )
        )

        # Transition components
        self.transition_dense = nn.Dense(self.hidden_dim, dtype=DTYPE)
        self.transition_resblock = self.resblock_fn(
            self.hidden_dim,
            norm_fn=None,
            hidden_N=self.hidden_N,
            activation=resolved_activation,
            use_swiglu=self.use_swiglu,
            dtype=DTYPE,
        )

        # Projection and Predictor components
        self.proj_mlp = MLP(self.hidden_dim, norm_fn=None, activation=resolved_activation)
        self.proj_dense = nn.Dense(self.projection_dim, dtype=HEAD_DTYPE)

        self.pred_mlp = MLP(self.hidden_dim, norm_fn=None, activation=resolved_activation)
        self.pred_dense = nn.Dense(self.projection_dim, dtype=HEAD_DTYPE)

    def states_to_latents(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        for resblock in self.embedding_resblocks:
            x = resblock(x, training)
        return x

    def latents_to_logits(self, x, training=False):
        for resblock in self.distances_resblocks:
            x = resblock(x, training)

        if isinstance(self.final_dense, nn.Dense):
            x = x.astype(HEAD_DTYPE)
            x = self.final_dense(x)
        else:
            x = self.final_dense(x, training)

        x = x.reshape(x.shape[:-1] + (self.action_size, self.categorial_n))
        return x

    def get_logits(self, x, training=False):
        latents = self.states_to_latents(x, training)
        return self.latents_to_logits(latents, training)

    def distance_and_latents(self, x, training=False):
        latents = self.states_to_latents(x, training)
        distances = self.latents_to_distances(latents, training)
        return distances, latents

    def transition(self, latents, actions, training=False):
        actions = jax.nn.one_hot(actions, self.path_action_size)
        x = jnp.concatenate([latents, actions], axis=-1)
        x = self.transition_dense(x)
        x = self.transition_resblock(x, training=training)
        return x

    def latents_to_projection(self, x, training=False):
        x = self.proj_mlp(x, training)
        x = self.proj_dense(x)
        return x

    def predict_ema_latents(self, x, training=False):
        x = self.pred_mlp(x, training)
        x = self.pred_dense(x)
        return x

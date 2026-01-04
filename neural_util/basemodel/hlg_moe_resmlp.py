from typing import Any

import jax
import jax.numpy as jnp
import optax
from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn

from neural_util.aqt_utils import build_aqt_dot_general
from neural_util.basemodel.base import DistanceHLGModel
from neural_util.modules import (
    DTYPE,
    HEAD_DTYPE,
    MLP,
    MoEPreActivationResBlock,
    MoEResBlock,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
    preactivation_MLP,
)


class HLGMoEResMLPModel(DistanceHLGModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: nn.Module = nn.LayerNorm
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.0
    min_capacity: int = 1
    router_temperature: float = 1.0
    router_noise_std: float = 0.0
    moe_aux_coef: float = 1e-2
    aqt_cfg: Any = None
    quant_mode: Any = None

    def setup(self):
        super().setup()

        aqt_dg = None
        if self.aqt_cfg is not None:
            mode = self.quant_mode if self.quant_mode is not None else aqt_flax.QuantMode.TRAIN
            aqt_dg = build_aqt_dot_general(self.aqt_cfg, mode)

        self.initial_mlp = (
            Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE, dot_general_cls=aqt_dg)
            if self.use_swiglu
            else MLP(
                self.initial_dim,
                norm_fn=self.norm_fn,
                activation=self.activation,
                dot_general_cls=aqt_dg,
            )
        )
        self.second_mlp = (
            (
                Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE, dot_general_cls=aqt_dg)
                if self.use_swiglu
                else MLP(
                    self.hidden_dim,
                    norm_fn=self.norm_fn,
                    activation=self.activation,
                    dot_general_cls=aqt_dg,
                )
            )
            if self.resblock_fn != PreActivationResBlock
            else nn.Dense(self.hidden_dim, dtype=DTYPE, dot_general_cls=aqt_dg)
        )

        block_cls = (
            MoEPreActivationResBlock if self.resblock_fn == PreActivationResBlock else MoEResBlock
        )

        # FLOPs parity: Scale down expert size by the number of active experts
        use_sparse = self.top_k is not None and self.top_k > 0 and self.top_k < self.num_experts
        active_experts = self.top_k if use_sparse else self.num_experts
        expert_node_size = max(1, (self.hidden_dim * self.hidden_node_multiplier) // active_experts)

        self.resblocks = [
            block_cls(
                expert_node_size,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                num_experts=self.num_experts,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                min_capacity=self.min_capacity,
                router_temperature=self.router_temperature,
                router_noise_std=self.router_noise_std,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.Res_N - self.tail_head_precision)
        ]
        self.tail_head_resblocks = [
            block_cls(
                expert_node_size,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                num_experts=self.num_experts,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                min_capacity=self.min_capacity,
                router_temperature=self.router_temperature,
                router_noise_std=self.router_noise_std,
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

    def _forward_logits(self, x, training=False, capture_aux=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        balance_losses = []
        gate_entropies = []
        router_z_losses = []
        if capture_aux:
            for resblock in self.resblocks:
                x, stats = resblock(x, training, capture_aux=True)
                balance_losses.append(stats["balance_loss"])
                gate_entropies.append(stats["gate_entropy"])
                router_z_losses.append(stats["router_z_loss"])
            for resblock in self.tail_head_resblocks:
                x, stats = resblock(x, training, capture_aux=True)
                balance_losses.append(stats["balance_loss"])
                gate_entropies.append(stats["gate_entropy"])
                router_z_losses.append(stats["router_z_loss"])
        else:
            for resblock in self.resblocks:
                x = resblock(x, training)
            for resblock in self.tail_head_resblocks:
                x = resblock(x, training)

        if isinstance(self.final_dense, nn.Dense):
            x = x.astype(HEAD_DTYPE)
            x = self.final_dense(x)
        else:
            x = self.final_dense(x, training)

        x = x.reshape(x.shape[0], self.action_size, self.categorial_n)

        if not capture_aux:
            return x, None

        if balance_losses:
            moe_balance_loss = jnp.mean(jnp.stack(balance_losses))
            moe_gate_entropy = jnp.mean(jnp.stack(gate_entropies))
            moe_router_z_loss = jnp.mean(jnp.stack(router_z_losses))
        else:
            moe_balance_loss = jnp.array(0.0, dtype=jnp.float32)
            moe_gate_entropy = jnp.array(0.0, dtype=jnp.float32)
            moe_router_z_loss = jnp.array(0.0, dtype=jnp.float32)
        return x, {
            "moe_balance_loss": moe_balance_loss,
            "moe_gate_entropy": moe_gate_entropy,
            "moe_router_z_loss": moe_router_z_loss,
        }

    def get_logits(self, x, training=False, capture_aux=False):
        if capture_aux:
            return self._forward_logits(x, training=training, capture_aux=True)
        logits, _ = self._forward_logits(x, training=training, capture_aux=False)
        return logits

    def __call__(self, x, training=False):
        logits = self.get_logits(x, training=training)
        return self.logit_to_values(logits)

    def train_loss(self, x, target, actions=None, **kwargs):
        categorial_bins, sigma = self.categorial_bins, self.sigma

        def f(target):
            cdf_evals = jax.scipy.special.erf((categorial_bins - target) / (jnp.sqrt(2) * sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(f)(target)
        logits_actions, moe_aux = self.get_logits(x, training=True, capture_aux=True)

        pred_actions = self.logit_to_values(logits_actions)

        if actions is None:
            logits = logits_actions.squeeze(1)
            pred = pred_actions.squeeze(1)
        else:
            logits = jnp.take_along_axis(logits_actions, actions[:, jnp.newaxis], axis=1).squeeze(1)
            pred = jnp.take_along_axis(pred_actions, actions[:, jnp.newaxis], axis=1).squeeze(1)

        sce = optax.softmax_cross_entropy(logits, target_probs)
        moe_balance_loss = moe_aux["moe_balance_loss"]
        # Include z-loss in auxiliary loss
        moe_router_z_loss = moe_aux["moe_router_z_loss"]
        moe_aux_loss = (moe_balance_loss + moe_router_z_loss) * self.moe_aux_coef
        loss = sce + moe_aux_loss
        aux = {
            "pred": pred,
            "sce": sce,
            "loss": loss,
            "diff": target - pred,
            "moe_balance_loss": moe_balance_loss,
            "moe_gate_entropy": moe_aux["moe_gate_entropy"],
            "moe_router_z_loss": moe_router_z_loss,
            "moe_aux_loss": moe_aux_loss,
        }
        return loss, aux

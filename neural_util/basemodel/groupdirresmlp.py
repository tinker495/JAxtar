from typing import Any

from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn

from neural_util.aqt_utils import build_aqt_dot_general
from neural_util.basemodel.base import DistanceGroupDIRModel
from neural_util.dtypes import DTYPE, HEAD_DTYPE
from neural_util.modules import (
    DEFAULT_NORM_FN,
    MLP,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
    preactivation_MLP,
)


class GroupDIRResMLPModel(DistanceGroupDIRModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: nn.Module = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    head_res_N: int = 1
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

        self.resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.Res_N - self.head_res_N)
        ]
        self.logit_head_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.head_res_N)
        ]
        self.logit_dense = (
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

        self.values_head_resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
                dot_general_cls=aqt_dg,
            )
            for _ in range(self.head_res_N)
        ]
        self.values_dense = (
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

    def get_logits_and_moe_values(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        for resblock in self.resblocks:
            x = resblock(x, training)

        x_logit = x
        x_values = x
        for logit_resblock in self.logit_head_resblocks:
            x_logit = logit_resblock(x_logit, training)
        for values_resblock in self.values_head_resblocks:
            x_values = values_resblock(x_values, training)

        if isinstance(self.logit_dense, nn.Dense):
            x_logit = x_logit.astype(HEAD_DTYPE)
            x_logit = self.logit_dense(x_logit)
        else:
            x_logit = self.logit_dense(x_logit, training)

        if isinstance(self.values_dense, nn.Dense):
            x_values = x_values.astype(HEAD_DTYPE)
            x_values = self.values_dense(x_values)
        else:
            x_values = self.values_dense(x_values, training)

        logits = x_logit.reshape(x_logit.shape[0], self.action_size, self.categorial_n)
        values = (
            x_values.reshape(x_values.shape[0], self.action_size, self.categorial_n)
            + self.categorial_centers
        )
        return logits, values

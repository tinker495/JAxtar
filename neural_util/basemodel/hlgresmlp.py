from typing import Any

from aqt.jax.v2.flax import aqt_flax
from flax import linen as nn

from neural_util.aqt_utils import build_aqt_dot_general
from neural_util.basemodel.base import DistanceHLGModel
from neural_util.dtypes import DTYPE, HEAD_DTYPE, PARAM_DTYPE
from neural_util.modules import (
    DEFAULT_NORM_FN,
    MLP,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
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
                activation=self.activation,
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
                    activation=self.activation,
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
                activation=self.activation,
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
                activation=self.activation,
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
        
        # Learnable Logit Scale (Temperature Adjustment)
        self.logit_scale = self.param(
            "logit_scale", 
            nn.initializers.constant(1.0), 
            (1,), 
            HEAD_DTYPE
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

        x = x.reshape(x.shape[0], self.action_size, self.categorial_n)
        
        # Apply Logit Scale
        x = x * self.logit_scale
        
        return x

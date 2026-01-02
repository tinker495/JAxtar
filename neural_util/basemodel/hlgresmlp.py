from flax import linen as nn

from neural_util.basemodel.base import DistanceHLGModel
from neural_util.basemodel.selfpredictive import SelfPredictiveDistanceHLGModel
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
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

    def setup(self):
        super().setup()

        self.initial_mlp = (
            Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE)
            if self.use_swiglu
            else MLP(self.initial_dim, norm_fn=self.norm_fn, activation=self.activation)
        )
        self.second_mlp = (
            (
                Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE)
                if self.use_swiglu
                else MLP(self.hidden_dim, norm_fn=self.norm_fn, activation=self.activation)
            )
            if self.resblock_fn != PreActivationResBlock
            else nn.Dense(self.hidden_dim, dtype=DTYPE)
        )

        self.resblocks = [
            self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
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
            )
            for _ in range(self.tail_head_precision)
        ]
        self.final_dense = (
            preactivation_MLP(self.action_size * self.categorial_n, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size * self.categorial_n,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
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


class SelfPredictiveHLGResMLPModel(HLGResMLPModel, SelfPredictiveDistanceHLGModel):
    def setup(self):
        super().setup()
        # Transition model components
        self.action_embedding = nn.Embed(self.action_size, self.hidden_dim, dtype=DTYPE)
        self.transition_resblock = self.resblock_fn(
            self.hidden_dim * self.hidden_node_multiplier,
            norm_fn=self.norm_fn,
            hidden_N=self.hidden_N,
            activation=self.activation,
            use_swiglu=self.use_swiglu,
            dtype=DTYPE,  # Keep in DTYPE (e.g. bfloat16/float32)
        )

        # Projection and Predictor components
        latent_dim = self.hidden_dim * self.hidden_node_multiplier
        self.proj_dense1 = nn.Dense(latent_dim, dtype=DTYPE)
        self.proj_norm1 = self.norm_fn()
        self.proj_dense2 = nn.Dense(latent_dim, dtype=DTYPE)
        self.pred_dense1 = nn.Dense(latent_dim, dtype=DTYPE)
        self.pred_norm1 = self.norm_fn()
        self.pred_dense2 = nn.Dense(latent_dim, dtype=DTYPE)

    def states_to_latents(self, x, training=False):
        x = self.initial_mlp(x, training)
        if isinstance(self.second_mlp, nn.Dense):
            x = self.second_mlp(x)
        else:
            x = self.second_mlp(x, training)

        for resblock in self.resblocks:
            x = resblock(x, training)
        return x

    def latents_to_logits(self, x, training=False):
        for resblock in self.tail_head_resblocks:
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
        action_embed = self.action_embedding(actions)  # (..., hidden_dim)
        if action_embed.ndim > latents.ndim:
            action_embed = action_embed.squeeze(-2)
        x = latents + action_embed
        x = self.transition_resblock(x, training=training)
        return x

    def latents_to_projection(self, x, training=False):
        x = self.proj_dense1(x)
        x = self.proj_norm1(x, training)
        x = self.activation(x)
        x = self.proj_dense2(x)
        return x

    def predict_ema_latents(self, x, training=False):
        x = self.pred_dense1(x)
        x = self.pred_norm1(x, training)
        x = self.activation(x)
        x = self.pred_dense2(x)
        return x

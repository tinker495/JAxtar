from flax import linen as nn

from neural_util.basemodel.base import DistanceModel
from neural_util.basemodel.selfpredictive import SelfPredictiveDistanceModel
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


class ResMLPModel(DistanceModel):
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
            preactivation_MLP(self.action_size, dtype=HEAD_DTYPE)
            if self.resblock_fn == PreActivationResBlock
            else nn.Dense(
                self.action_size,
                dtype=HEAD_DTYPE,
                kernel_init=nn.initializers.normal(stddev=0.01),
            )
        )

    def __call__(self, x, training=False):
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
        return x


class SelfPredictiveResMLPModel(SelfPredictiveDistanceModel):
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1
    tail_head_precision: int = 0

    def setup(self):
        # Re-implementing using setup for cleaner separation.
        if self.use_swiglu:
            self.initial_layer = Swiglu(self.initial_dim, norm_fn=self.norm_fn, dtype=DTYPE)
            if self.resblock_fn != PreActivationResBlock:
                self.second_layer = Swiglu(self.hidden_dim, norm_fn=self.norm_fn, dtype=DTYPE)
            else:
                self.second_layer = nn.Dense(self.hidden_dim, dtype=DTYPE)
        else:
            # Need to compose these manually if not using a block
            self.initial_dense = nn.Dense(self.initial_dim, dtype=DTYPE)
            self.initial_norm = self.norm_fn()
            self.second_dense = nn.Dense(self.hidden_dim, dtype=DTYPE)
            if self.resblock_fn != PreActivationResBlock:
                self.second_norm = self.norm_fn()

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
        if self.resblock_fn == PreActivationResBlock:
            self.final_norm = self.norm_fn()

        self.final_dense = nn.Dense(
            self.action_size, dtype=HEAD_DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )

        # Transition components
        self.action_embedding = nn.Embed(self.action_size, self.hidden_dim, dtype=DTYPE)
        self.transition_resblock = self.resblock_fn(
            self.hidden_dim * self.hidden_node_multiplier,
            norm_fn=self.norm_fn,
            hidden_N=self.hidden_N,
            activation=self.activation,
            use_swiglu=self.use_swiglu,
            dtype=DTYPE,
        )

        # Projection and Predictor components
        latent_dim = self.hidden_dim * self.hidden_node_multiplier
        self.proj_dense1 = nn.Dense(latent_dim, dtype=DTYPE)
        self.proj_norm1 = self.norm_fn()
        self.proj_dense2 = nn.Dense(latent_dim, dtype=DTYPE)

        self.pred_dense1 = nn.Dense(latent_dim, dtype=DTYPE)
        self.pred_norm1 = self.norm_fn()
        self.pred_dense2 = nn.Dense(latent_dim, dtype=DTYPE)

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

    def states_to_latents(self, x, training=False):
        if self.use_swiglu:
            x = self.initial_layer(x, training)
            if self.resblock_fn != PreActivationResBlock:
                x = self.second_layer(x, training)
            else:
                x = self.second_layer(x)
        else:
            x = self.initial_dense(x)
            x = self.initial_norm(x, training)
            x = self.activation(x)
            x = self.second_dense(x)
            if self.resblock_fn != PreActivationResBlock:
                x = self.second_norm(x, training)
                x = self.activation(x)

        for resblock in self.resblocks:
            x = resblock(x, training)
        return x

    def latents_to_distances(self, x, training=False):
        for resblock in self.tail_head_resblocks:
            x = resblock(x, training)

        if self.resblock_fn == PreActivationResBlock:
            x = self.final_norm(x, training)
            x = self.activation(x)

        x = x.astype(HEAD_DTYPE)
        x = self.final_dense(x)
        return x

    def __call__(self, x, training=False):
        latents = self.states_to_latents(x, training)
        return self.latents_to_distances(latents, training)

    def distance_and_latents(self, x, training=False):
        latents = self.states_to_latents(x, training)
        distances = self.latents_to_distances(latents, training)
        return distances, latents

    def transition(self, latents, actions, training=False):
        action_embed = self.action_embedding(actions)
        x = latents + action_embed
        x = self.transition_resblock(x, training=training)
        return x

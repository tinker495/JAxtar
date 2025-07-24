import jax
import jax.numpy as jnp
from flax import linen as nn

from neural_util.modules import DEFAULT_NORM_FN, DTYPE, ResBlock, conditional_dummy_norm
from neural_util.spr_modules import Encoder, ProjectionHead, TransitionModel
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class QHead(nn.Module):
    """Predicts the Q-values from a latent representation."""

    action_size: int
    Res_N: int = 2
    latent_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        for _ in range(self.Res_N):
            x = ResBlock(self.latent_dim, norm_fn=self.norm_fn, activation=self.activation)(
                x, training
            )
        x = nn.Dense(
            self.action_size, dtype=DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        _ = conditional_dummy_norm(x, self.norm_fn, training)
        return x


class SPRQModel(nn.Module):
    """
    The full SPR model for Q-value prediction.
    """

    action_size: int
    Res_N_Encoder: int = 2
    Res_N_Q: int = 2
    latent_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    def setup(self) -> None:
        self.encoder = Encoder(
            Res_N=self.Res_N_Encoder,
            latent_dim=self.latent_dim,
            norm_fn=self.norm_fn,
            activation=self.activation,
            name="encoder",
        )
        self.q_head = QHead(
            action_size=self.action_size,
            Res_N=self.Res_N_Q,
            latent_dim=self.latent_dim,
            norm_fn=self.norm_fn,
            activation=self.activation,
            name="q_head",
        )
        self.projection_head = ProjectionHead(
            output_dim=self.latent_dim, activation=self.activation, name="projection_head"
        )
        self.transition_model = TransitionModel(
            action_size=self.action_size,
            latent_dim=self.latent_dim,
            activation=self.activation,
            name="transition_model",
        )
        self.predicton_head = ProjectionHead(
            output_dim=self.latent_dim, activation=self.activation, name="prediction_head"
        )

    @nn.compact
    def __call__(self, x, training=False):
        # Online network
        latent_z = self.encoder(x, training)  # (batch_size, latent_dim)
        q_values = self.q_head(latent_z, training)  # (batch_size, action_size)

        projected_p = self.projection_head(latent_z)  # (batch_size, latent_dim)
        transition = self.transition_model(latent_z)  # (batch_size, action_size, latent_dim)
        transition = transition[:, 0]  # (batch_size, latent_dim)

        predicted_next_p = self.predicton_head(transition)  # (batch_size, latent_dim)

        return q_values, projected_p, predicted_next_p

    def get_q(self, x, training=False):
        latent_z = self.encoder(x, training)
        return self.q_head(latent_z, training)

    def get_projected_p(self, x, training=False):
        latent_z = self.encoder(x, training)
        return self.projection_head(latent_z)

    def get_q_and_predicted_next_p(self, x, actions, training=False):
        latent_z = self.encoder(x, training)
        q_values = self.q_head(latent_z, training)
        q_values_at_actions = jnp.take_along_axis(q_values, actions[:, jnp.newaxis], axis=1)
        transition = self.transition_model(latent_z)
        transition = jnp.take_along_axis(
            transition, actions[:, jnp.newaxis, jnp.newaxis], axis=1
        ).squeeze(1)
        projected_p = self.projection_head(transition)
        predicted_next_p = self.predicton_head(projected_p)
        return q_values_at_actions, predicted_next_p


class SPRNeuralQFunction(NeuralQFunctionBase):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=SPRQModel, **kwargs)

    def batched_param_q_value(self, params, solve_config, current):
        x = jax.vmap(self.pre_process, in_axes=(None, 0))(solve_config, current)
        q_values = self.model.apply(
            params, x, training=False, mutable=["batch_stats"], method=self.model.get_q
        )[0]
        return self.post_process(q_values)

    def param_q_value(self, params, solve_config, current):
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        q_values = self.model.apply(
            params, x, training=False, mutable=["batch_stats"], method=self.model.get_q
        )[0]
        return self.post_process(q_values)

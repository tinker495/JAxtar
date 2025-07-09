import jax
import jax.numpy as jnp
from flax import linen as nn

from neural_util.modules import DTYPE, ResBlock, conditional_dummy_norm
from neural_util.spr_modules import Encoder, ProjectionHead, TransitionModel
from qfunction.neuralq.neuralq_base import NeuralQFunctionBase


class QHead(nn.Module):
    """Predicts the Q-values from a latent representation."""

    action_size: int
    Res_N: int = 2
    latent_dim: int = 1000

    @nn.compact
    def __call__(self, x, training=False):
        for _ in range(self.Res_N):
            x = ResBlock(self.latent_dim)(x, training)
        x = nn.Dense(
            self.action_size, dtype=DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        _ = conditional_dummy_norm(x, training)
        return x


class SPRQModel(nn.Module):
    """
    The full SPR model for Q-value prediction.
    """

    action_size: int
    Res_N_Encoder: int = 2
    Res_N_Q: int = 2
    latent_dim: int = 1000

    @nn.compact
    def __call__(self, x, training=False):
        # Online network
        encoder = Encoder(Res_N=self.Res_N_Encoder, latent_dim=self.latent_dim, name="encoder")
        q_head = QHead(
            action_size=self.action_size,
            Res_N=self.Res_N_Q,
            latent_dim=self.latent_dim,
            name="q_head",
        )
        projection_head = ProjectionHead(output_dim=self.latent_dim, name="projection_head")
        transition_model = TransitionModel(
            action_size=self.action_size, latent_dim=self.latent_dim, name="transition_model"
        )

        # Forward pass
        latent_z = encoder(x, training)
        q_values = q_head(latent_z, training)

        projected_p = projection_head(latent_z)
        predicted_next_p_all_actions = transition_model(latent_z)

        return q_values, projected_p, predicted_next_p_all_actions


class SPRNeuralQFunction(NeuralQFunctionBase):
    def __init__(self, puzzle, **kwargs):
        super().__init__(puzzle, model=SPRQModel, **kwargs)

    def batched_param_q_value(self, params, solve_config, current):
        x = jax.vmap(self.pre_process, in_axes=(None, 0))(solve_config, current)
        q_values, _, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])[0]
        return self.post_process(q_values)

    def param_q_value(self, params, solve_config, current):
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        q_values, _, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])[0]
        return self.post_process(q_values)

import jax
import jax.numpy as jnp
from flax import linen as nn
from puxle import Puzzle

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE, ResBlock, conditional_dummy_norm
from neural_util.spr_modules import Encoder, ProjectionHead, TransitionModel


class DistHead(nn.Module):
    """Predicts the heuristic value from a latent representation."""

    Res_N: int = 2
    latent_dim: int = 1000

    @nn.compact
    def __call__(self, x, training=False):
        for _ in range(self.Res_N):
            x = ResBlock(self.latent_dim)(x, training)
        x = nn.Dense(1, dtype=DTYPE, kernel_init=nn.initializers.normal(stddev=0.01))(x)
        _ = conditional_dummy_norm(x, training)
        return x


class SPRHeuristicModel(nn.Module):
    """
    The full SPR model for heuristic prediction, including encoder, heads,
    and transition model.
    """

    action_size: int
    Res_N_Encoder: int = 2
    Res_N_Dist: int = 2
    latent_dim: int = 1000

    def setup(self) -> None:
        self.encoder = Encoder(Res_N=self.Res_N_Encoder, latent_dim=self.latent_dim, name="encoder")
        self.dist_head = DistHead(
            Res_N=self.Res_N_Dist, latent_dim=self.latent_dim, name="dist_head"
        )
        self.projection_head = ProjectionHead(output_dim=self.latent_dim, name="projection_head")
        self.transition_model = TransitionModel(
            action_size=self.action_size, latent_dim=self.latent_dim, name="transition_model"
        )
        self.predicton_head = ProjectionHead(output_dim=self.latent_dim, name="prediction_head")

    @nn.compact
    def __call__(self, x, training=False):
        # Online network
        latent_z = self.encoder(x, training)
        heuristic = self.dist_head(latent_z, training)

        projected_p = self.projection_head(latent_z)
        transition = self.transition_model(latent_z)
        transition = transition[:, 0]

        predicted_next_p = self.predicton_head(transition)

        return heuristic, projected_p, predicted_next_p

    def get_heuristic(self, x, training=False):
        latent_z = self.encoder(x, training)
        return self.dist_head(latent_z, training)

    def get_projected_p(self, x, training=False):
        latent_z = self.encoder(x, training)
        return self.projection_head(latent_z)

    def get_heuristic_and_predicted_next_p(self, x, actions, training=False):
        latent_z = self.encoder(x, training)
        heuristic = self.dist_head(latent_z, training)
        transition = self.transition_model(latent_z)
        transition = jnp.take_along_axis(
            transition, actions[:, jnp.newaxis, jnp.newaxis], axis=1
        ).squeeze(1)
        projected_p = self.projection_head(transition)
        predicted_next_p = self.predicton_head(projected_p)
        return heuristic, predicted_next_p


class SPRNeuralHeuristic(NeuralHeuristicBase):
    def __init__(self, puzzle: Puzzle, **kwargs):
        # Get action size for the model
        dummy_solve_config = puzzle.SolveConfig.default()
        dummy_current = puzzle.State.default()
        action_size = len(puzzle.get_neighbours(dummy_solve_config, dummy_current))

        # Add action_size to kwargs for the model
        model_kwargs = kwargs.copy()
        model_kwargs["action_size"] = action_size

        super().__init__(puzzle, model=SPRHeuristicModel, **model_kwargs)

    def batched_param_distance(self, params, solve_config, current):
        x = jax.vmap(self.pre_process, in_axes=(None, 0))(solve_config, current)
        heuristic = self.model.apply(
            params, x, training=False, mutable=["batch_stats"], method=self.model.get_heuristic
        )[0]
        return self.post_process(heuristic)

    def param_distance(self, params, solve_config, current):
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        heuristic = self.model.apply(
            params, x, training=False, mutable=["batch_stats"], method=self.model.get_heuristic
        )[0]
        return self.post_process(heuristic)

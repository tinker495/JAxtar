import os
import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction

from .modules import BatchNorm, ResBlock
from .util import download_model, is_model_downloaded


class Projector(nn.Module):
    projection_dim: int

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = nn.Dense(self.projection_dim)(x)
        return x


class Predictor(nn.Module):
    projection_dim: int

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.projection_dim)(x)
        return x


class DefaultModel(nn.Module):
    action_size: int = 4
    projection_dim: int = 128
    use_solve_config: bool = True

    def setup(self):
        if self.use_solve_config:
            self.solve_config_projector = Projector(projection_dim=self.projection_dim)

        self.state_projector = Projector(projection_dim=self.projection_dim)
        self.predictor = Predictor(projection_dim=self.projection_dim)
        self.distance_bias = self.param("distance_bias", nn.initializers.zeros, (1,))
        self.distance_scale = self.param("distance_scale", nn.initializers.ones, (1,))
        self.action_projection = self.param(
            "action_projection", nn.initializers.zeros, (1, self.action_size, self.projection_dim)
        )  # [action_size, projection_dim]

    def __call__(
        self, preprocessed_target: chex.Array, preprocessed_current: chex.Array, training=False
    ):
        if self.use_solve_config:
            target_projection = self.solve_config_projector(
                preprocessed_target, training
            )  # [batch_size, projection_dim]
        else:
            target_projection = self.state_projector(
                preprocessed_target, training
            )  # [batch_size, projection_dim]
        current_projection = self.state_projector(
            preprocessed_current, training
        )  # [batch_size, projection_dim]
        actioned_projection = (
            jnp.expand_dims(current_projection, axis=1) + self.action_projection
        )  # [batch_size, action_size, projection_dim]
        dot_product = jnp.einsum(
            "bd, bad -> ba", target_projection, actioned_projection
        )  # [batch_size, action_size]
        distances = (
            self.distance_bias + self.distance_scale * dot_product
        )  # [batch_size, action_size]
        return distances  # [batch_size, action_size]

    def state_distance(
        self, preprocessed_target: chex.Array, preprocessed_current: chex.Array, training=False
    ):
        target_projection = self.state_projector(
            preprocessed_target, training
        )  # [batch_size, projection_dim]
        current_projection = self.state_projector(
            preprocessed_current, training
        )  # [batch_size, projection_dim]
        actioned_projection = (
            jnp.expand_dims(current_projection, axis=1) + self.action_projection
        )  # [batch_size, action_size, projection_dim]
        dot_product = jnp.einsum(
            "bd, bad -> ba", target_projection, actioned_projection
        )  # [batch_size, action_size]
        distances = (
            self.distance_bias + self.distance_scale * dot_product
        )  # [batch_size, action_size]
        return distances  # [batch_size, action_size]

    def state_similarity(self, state1: chex.Array, state2: chex.Array, training=False):
        projection1 = self.state_projector(state1, training)  # [batch_size, projection_dim]
        projection2 = self.state_projector(state2, training)  # [batch_size, projection_dim]
        prediction = self.predictor(projection1, training)  # [batch_size, projection_dim]
        dot_product = jnp.einsum("bd, bd -> b", prediction, projection2)  # [batch_size]
        # Normalize the vectors for cosine similarity
        norm1 = jnp.sqrt(jnp.sum(prediction**2, axis=1))  # [batch_size]
        norm2 = jnp.sqrt(jnp.sum(projection2**2, axis=1))  # [batch_size]
        # Avoid division by zero
        denominator = jnp.maximum(norm1 * norm2, 1e-12)  # [batch_size]
        # Calculate cosine similarity
        cos_similarity = dot_product / denominator  # [batch_size]
        return cos_similarity  # [batch_size]


class NeuralQFunctionBase(QFunction):
    def __init__(
        self,
        puzzle: Puzzle,
        projection_dim: int = 128,
        model: nn.Module = DefaultModel,
        init_params: bool = True,
    ):
        self.puzzle = puzzle
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        self.action_size = self.puzzle.get_neighbours(dummy_solve_config, dummy_current)[0].shape[
            0
        ][0]
        self.model = model(
            self.action_size,
            projection_dim=projection_dim,
            use_solve_config=not self.puzzle.only_target,
        )
        if init_params:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
        )

    @classmethod
    def load_model(cls, puzzle: Puzzle, path: str):

        try:
            if not is_model_downloaded(path):
                download_model(path)
            with open(path, "rb") as f:
                params = pickle.load(f)
            qfunc = cls(puzzle, init_params=False)
            dummy_solve_config = puzzle.SolveConfig.default()
            dummy_current = puzzle.State.default()
            qfunc.model.apply(
                params,
                jnp.expand_dims(qfunc.pre_process(dummy_solve_config, dummy_current), axis=0),
                training=False,
            )  # check if the params are compatible with the model
            qfunc.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            qfunc = cls(puzzle)
        return qfunc

    def save_model(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def batched_q_value(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_q_value(self.params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = jax.vmap(self.pre_process, in_axes=(None, 0))(solve_config, current)
        x, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])
        x = self.post_process(x)
        return x

    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return self.param_q_value(self.params, solve_config, current)[0]

    def param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        x, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])
        return self.post_process(x)

    @abstractmethod
    def pre_process(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pass

    def post_process(self, x: chex.Array) -> float:
        """
        This function should return the post-processed distance.
        """
        return x

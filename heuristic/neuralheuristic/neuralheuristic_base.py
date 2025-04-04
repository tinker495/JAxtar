import os
import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from heuristic.heuristic_base import Heuristic
from puzzle.puzzle_base import Puzzle

from .modules import BatchNorm, LayerNorm, ResBlock, cosine_similarity
from .util import download_model, is_model_downloaded


class Projector(nn.Module):
    projection_dim: int

    @nn.compact
    def __call__(self, x, training=False):
        _ = BatchNorm(x, training)  # for dummy batchnorm
        x = nn.Dense(1000)(x)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = LayerNorm(x, training)
        x = nn.Dense(self.projection_dim)(x)
        return x


class Predictor(nn.Module):
    projection_dim: int

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = LayerNorm(x, training)
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = LayerNorm(x, training)
        x = nn.Dense(self.projection_dim)(x)
        return x


class DefaultModel(nn.Module):
    projection_dim: int = 128

    def setup(self):
        self.solve_config_projector = Projector(projection_dim=self.projection_dim)
        self.state_projector = Projector(projection_dim=self.projection_dim)
        self.predictor = Predictor(projection_dim=self.projection_dim)
        self.distance_conv = nn.Dense(1, kernel_init=nn.initializers.zeros)

    def __call__(
        self,
        preprocessed_solve_config: chex.Array,
        preprocessed_state: chex.Array,
        training=False,
    ):
        a = self.solve_config_distance(preprocessed_solve_config, preprocessed_state, training)
        b = self.state_similarity(preprocessed_state, preprocessed_state, training)
        return a, b

    def solve_config_distance(
        self, preprocessed_target: chex.Array, preprocessed_current: chex.Array, training=False
    ):
        target_projection = self.solve_config_projector(
            preprocessed_target, training
        )  # [batch_size, projection_dim]
        current_projection = self.state_projector(
            preprocessed_current, training
        )  # [batch_size, projection_dim]
        dot_product = jnp.einsum(
            "bd, bd -> b", target_projection, current_projection
        )  # [batch_size]
        concat = jnp.concatenate(
            [target_projection, current_projection, dot_product[:, jnp.newaxis]], axis=1
        )
        distance = self.distance_conv(concat)  # [batch_size, 1]
        return distance  # [batch_size, 1]

    def get_solve_config_projection(self, preprocessed_solve_config: chex.Array, training=False):
        return self.solve_config_projector(preprocessed_solve_config, training)

    def get_state_projection(self, preprocessed_state: chex.Array, training=False):
        return self.state_projector(preprocessed_state, training)

    def distance_from_projection(
        self, target_projection: chex.Array, current_projection: chex.Array, training=False
    ):
        dot_product = jnp.einsum(
            "bd, bd -> b", target_projection, current_projection
        )  # [batch_size]
        concat = jnp.concatenate(
            [target_projection, current_projection, dot_product[:, jnp.newaxis]], axis=1
        )
        distance = self.distance_conv(concat)  # [batch_size, 1]
        return distance  # [batch_size, 1]

    def state_similarity(self, state1: chex.Array, state2: chex.Array, training=False):
        projection1 = self.state_projector(state1, training)  # [batch_size, projection_dim]
        projection2 = self.state_projector(state2, training)  # [batch_size, projection_dim]
        prediction1 = self.predictor(projection1, training)  # [batch_size, projection_dim]
        prediction2 = self.predictor(projection2, training)  # [batch_size, projection_dim]
        cos_similarity1 = cosine_similarity(
            prediction1, jax.lax.stop_gradient(projection2)
        )  # [batch_size]
        cos_similarity2 = cosine_similarity(
            prediction2, jax.lax.stop_gradient(projection1)
        )  # [batch_size]
        return cos_similarity1, cos_similarity2


class NeuralHeuristicBase(Heuristic):
    def __init__(
        self,
        puzzle: Puzzle,
        projection_dim: int = 128,
        model: nn.Module = DefaultModel,
        init_params: bool = True,
    ):
        self.puzzle = puzzle
        self.model = model(projection_dim=projection_dim)
        if init_params:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        dummy_solve_config = jnp.expand_dims(
            self.pre_process_solve_config(dummy_solve_config), axis=0
        )
        dummy_current = jnp.expand_dims(self.pre_process_state(dummy_current), axis=0)
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            dummy_solve_config,
            dummy_current,
        )

    @classmethod
    def load_model(cls, puzzle: Puzzle, path: str):

        try:
            if not is_model_downloaded(path):
                download_model(path)
            with open(path, "rb") as f:
                params = pickle.load(f)
            heuristic = cls(puzzle, init_params=False)
            dummy_solve_config = puzzle.SolveConfig.default()
            dummy_current = puzzle.State.default()
            dummy_solve_config = jnp.expand_dims(
                heuristic.pre_process_solve_config(dummy_solve_config), axis=0
            )
            dummy_current = jnp.expand_dims(heuristic.pre_process_state(dummy_current), axis=0)
            heuristic.model.apply(
                params,
                dummy_solve_config,
                dummy_current,
                training=False,
            )  # check if the params are compatible with the model
            heuristic.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            heuristic = cls(puzzle)
        return heuristic

    def save_model(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def batched_distance(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_distance(self.params, solve_config, current)

    def batched_param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        preprocessed_solve_config = self.pre_process_solve_config(solve_config)  # [...]
        preprocessed_current = jax.vmap(self.pre_process_state)(current)  # [batch_size, ...]
        target_projection, _ = self.model.apply(
            params,
            jnp.expand_dims(preprocessed_solve_config, axis=0),
            training=False,
            mutable=["batch_stats"],
            method=self.model.get_solve_config_projection,
        )  # [1, projection_dim]
        current_projection, _ = self.model.apply(
            params,
            preprocessed_current,
            training=False,
            mutable=["batch_stats"],
            method=self.model.get_state_projection,
        )  # [batch_size, projection_dim]
        target_projection = jnp.tile(target_projection, (preprocessed_current.shape[0], 1))
        value, _ = self.model.apply(
            params,
            target_projection,
            current_projection,
            training=False,
            mutable=["batch_stats"],
            method=self.model.distance_from_projection,
        )  # [batch_size, 1]
        return self.post_process(value)

    def distance(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return float(self.param_distance(self.params, solve_config, current)[0])

    def param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        preprocessed_solve_config = self.pre_process_solve_config(solve_config)[
            jnp.newaxis, :
        ]  # [1, ...]
        preprocessed_current = self.pre_process_state(current)[jnp.newaxis, :]
        value, _ = self.model.apply(
            params,
            preprocessed_solve_config,
            preprocessed_current,
            training=False,
            mutable=["batch_stats"],
            method=self.model.solve_config_distance,
        )  # [1, 1]
        return self.post_process(value)

    def pre_process_solve_config(self, solve_config: Puzzle.SolveConfig) -> chex.Array:
        """
        This function should return the pre-processed solve config.
        """
        assert (
            self.puzzle.only_target
        ), "This config is for only target condition, you should redefine this function for your puzzle"
        target_state = solve_config.TargetState
        return self.pre_process_state(target_state)

    @abstractmethod
    def pre_process_state(self, state: Puzzle.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pass

    def post_process(self, x: chex.Array) -> float:
        """
        This function should return the post-processed distance.
        """
        return x.squeeze(1)

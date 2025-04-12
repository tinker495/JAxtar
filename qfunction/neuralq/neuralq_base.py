import os
import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from neural_util.modules import DTYPE, BatchNorm, LayerNorm, ResBlock
from neural_util.util import download_model, is_model_downloaded
from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


class Projector(nn.Module):
    projection_dim: int

    @nn.compact
    def __call__(self, x, training=False):
        _ = BatchNorm(x, training)  # for dummy batchnorm
        x = nn.Dense(1000)(x)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = LayerNorm(x, training)
        x = nn.Dense(self.projection_dim, dtype=DTYPE)(x)
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
    action_size: int = 4
    projection_dim: int = 128

    def setup(self):
        self.solve_config_projector = Projector(projection_dim=self.projection_dim)
        self.state_projector = Projector(projection_dim=self.projection_dim)
        self.predictor = Predictor(projection_dim=self.projection_dim)
        self.action_projection = self.param(
            "action_projection",
            nn.initializers.he_normal(),
            (1, self.action_size, self.projection_dim),
        )  # [1, action_size, projection_dim]
        self.distance_conv = nn.Dense(self.action_size, kernel_init=nn.initializers.zeros)

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
        actioned_projection = (
            jnp.expand_dims(current_projection, axis=1) + self.action_projection
        )  # [batch_size, action_size, projection_dim]
        dot_product = jnp.einsum(
            "bd, bad -> ba", target_projection, actioned_projection
        )  # [batch_size, action_size]
        flatten_action_projection = jnp.reshape(
            actioned_projection, (-1, self.projection_dim * self.action_size)
        )
        concat = jnp.concatenate(
            [target_projection, flatten_action_projection, dot_product], axis=1
        )
        distances = self.distance_conv(concat)  # [batch_size, action_size]
        return distances  # [batch_size, action_size]

    def get_solve_config_projection(self, preprocessed_solve_config: chex.Array, training=False):
        return self.solve_config_projector(preprocessed_solve_config, training)

    def get_state_projection(self, preprocessed_state: chex.Array, training=False):
        return self.state_projector(preprocessed_state, training)

    def distance_from_projection(
        self, target_projection: chex.Array, current_projection: chex.Array, training=False
    ):
        actioned_projection = (
            jnp.expand_dims(current_projection, axis=1) + self.action_projection
        )  # [batch_size, action_size, projection_dim]
        dot_product = jnp.einsum(
            "bd, bad -> ba", target_projection, actioned_projection
        )  # [batch_size, action_size]
        flatten_action_projection = jnp.reshape(
            actioned_projection, (-1, self.projection_dim * self.action_size)
        )
        concat = jnp.concatenate(
            [target_projection, flatten_action_projection, dot_product], axis=1
        )
        distance = self.distance_conv(concat)  # [batch_size, action_size]
        return distance  # [batch_size, action_size]


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
        )
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
            qfunc = cls(puzzle, init_params=False)
            dummy_solve_config = puzzle.SolveConfig.default()
            dummy_current = puzzle.State.default()
            dummy_solve_config = jnp.expand_dims(
                qfunc.pre_process_solve_config(dummy_solve_config), axis=0
            )
            dummy_current = jnp.expand_dims(qfunc.pre_process_state(dummy_current), axis=0)
            qfunc.model.apply(
                params,
                dummy_solve_config,
                dummy_current,
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
        q_value, _ = self.model.apply(
            params,
            target_projection,
            current_projection,
            training=False,
            mutable=["batch_stats"],
            method=self.model.distance_from_projection,
        )  # [batch_size, action_size]
        return q_value

    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return self.param_q_value(self.params, solve_config, current)[0]

    def param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        preprocessed_solve_config = self.pre_process_solve_config(solve_config)[
            jnp.newaxis, :
        ]  # [1, ...]
        preprocessed_current = self.pre_process_state(current)[jnp.newaxis, :]
        q_value, _ = self.model.apply(
            params,
            preprocessed_solve_config,
            preprocessed_current,
            training=False,
            mutable=["batch_stats"],
            method=self.model.solve_config_distance,
        )  # [1, action_size]
        return self.post_process(q_value)

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
        return x

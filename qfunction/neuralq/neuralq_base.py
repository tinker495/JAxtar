import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction

from .util import download_model, is_model_downloaded


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9)(x, use_running_average=not training)


def AvgL1Norm(x):
    return x / jnp.mean(jnp.abs(x), axis=-1, keepdims=True)


# Residual Block
class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size)(x0)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        x = BatchNorm(x, training)
        return nn.relu(x + x0)


class DistanceModel(nn.Module):
    action_size: int = 4

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = nn.Dense(self.action_size)(x)
        return x


class Projector(nn.Module):
    project_dim: int = 512

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.project_dim)(x)
        return x


class Predictor(nn.Module):
    predict_dim: int = 512

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(1000)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.predict_dim)(x)
        return x


class NeuralQFunctionBase(QFunction):
    def __init__(self, puzzle: Puzzle, model: nn.Module = DistanceModel, init_params: bool = True):
        self.puzzle = puzzle
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        self.action_size = self.puzzle.get_neighbours(dummy_solve_config, dummy_current)[0].shape[
            0
        ][0]

        class TotalModel(nn.Module):
            distance_model: DistanceModel
            solve_config_projector: Projector = Projector()
            state_projector: Projector = Projector()
            predictor: Predictor = Predictor()

            @nn.compact
            def __call__(self, solve_config, current, training=False):
                solve_config_project = self.solve_config_projector(solve_config, training)
                state_project = self.state_projector(current, training)
                state_predict = self.predictor(state_project, training)
                stacked_project = jnp.concatenate([solve_config_project, state_project], axis=-1)
                stacked_project = AvgL1Norm(stacked_project)
                # stacked_raw = jnp.concatenate([solve_config, current], axis=-1)
                # stacked_project_raw = jnp.concatenate([stacked_project, stacked_raw], axis=-1)
                distance = self.distance_model(stacked_project, training)
                return distance, state_predict

            def distance(self, solve_config, current, training=False):
                solve_config_project = self.solve_config_projector(solve_config, training)
                state_project = self.state_projector(current, training)
                stacked_project = jnp.concatenate([solve_config_project, state_project], axis=-1)
                stacked_project = AvgL1Norm(stacked_project)
                # stacked_raw = jnp.concatenate([solve_config, current], axis=-1)
                # stacked_project_raw = jnp.concatenate([stacked_project, stacked_raw], axis=-1)
                distance = self.distance_model(stacked_project, training)
                return distance

            def project_solve_config(self, solve_config, training=False):
                project = self.solve_config_projector(solve_config, training)
                return project

            def project_state(self, state, training=False):
                project = self.state_projector(state, training)
                return project

            def predict(self, project, training=False):
                predict = self.predictor(project, training)
                return predict

            def train_info(self, solve_config, state, next_state, training=True):
                distance, state_predict = self.__call__(solve_config, state, training)
                next_state_project = self.project_state(next_state, training)
                return distance, state_predict, next_state_project

        self.model = TotalModel(distance_model=model(self.action_size))

        if init_params:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(self.solve_config_pre_process(dummy_solve_config), axis=0),
            jnp.expand_dims(self.state_pre_process(dummy_current), axis=0),
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
                jnp.expand_dims(qfunc.solve_config_pre_process(dummy_solve_config), axis=0),
                jnp.expand_dims(qfunc.state_pre_process(dummy_current), axis=0),
                training=False,
            )  # check if the params are compatible with the model
            qfunc.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            qfunc = cls(puzzle)
        return qfunc

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def batched_q_value(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_q_value(self.params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        solve_config_preprocessed = self.solve_config_pre_process(solve_config)
        state_preprocessed = jax.vmap(self.state_pre_process)(current)
        solve_config_preprocessed = jnp.tile(
            jnp.expand_dims(solve_config_preprocessed, axis=0), (state_preprocessed.shape[0], 1)
        )
        x, _ = self.model.apply(
            params,
            solve_config_preprocessed,
            state_preprocessed,
            training=False,
            mutable=["batch_stats"],
            method=self.model.distance,
        )
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
        solve_config_preprocessed = jnp.expand_dims(
            self.solve_config_pre_process(solve_config), axis=0
        )
        state_preprocessed = jnp.expand_dims(self.state_pre_process(current), axis=0)
        x, _ = self.model.apply(
            params,
            solve_config_preprocessed,
            state_preprocessed,
            training=False,
            mutable=["batch_stats"],
            method=self.model.distance,
        )
        return self.post_process(x)

    @abstractmethod
    def solve_config_pre_process(self, solve_config: Puzzle.SolveConfig) -> chex.Array:
        """
        This function should return the pre-processed solve_config.
        """
        pass

    @abstractmethod
    def state_pre_process(self, state: Puzzle.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pass

    def post_process(self, x: chex.Array) -> float:
        """
        This function should return the post-processed distance.
        """
        return x

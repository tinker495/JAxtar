import pickle
from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from puzzle.puzzle_base import Puzzle


# Residual Block
class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size)(x0)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        return nn.relu(x + x0)


class DefaultModel(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(5000)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = ResBlock(1000)(x, training)
        x = nn.Dense(1)(x)
        return x


class NeuralHeuristicBase(ABC):
    def __init__(self, puzzle: Puzzle, model: nn.Module = DefaultModel(), init_params: bool = True):
        self.puzzle = puzzle
        self.model = model
        if init_params:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_current = self.puzzle.State.default()
        dummy_target = self.puzzle.State.default()
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(self.pre_process(dummy_current, dummy_target), axis=0),
        )

    @classmethod
    def load_model(cls, puzzle: Puzzle, path: str):

        try:
            with open(path, "rb") as f:
                params = pickle.load(f)
            heuristic = cls(puzzle, init_params=False)
            dummy_current = puzzle.State.default()
            dummy_target = puzzle.State.default()
            heuristic.model.apply(
                params,
                jnp.expand_dims(heuristic.pre_process(dummy_current, dummy_target), axis=0),
                training=False,
            )  # check if the params are compatible with the model
            heuristic.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            heuristic = cls(puzzle)
        return heuristic

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def batched_distance(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        return self.batched_param_distance(self.params, current, target)

    def batched_param_distance(
        self, params, current: Puzzle.State, target: Puzzle.State
    ) -> chex.Array:
        x = jax.vmap(self.pre_process, in_axes=(0, None))(current, target)
        x, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])
        x = self.post_process(x)
        return x

    def distance(self, current: Puzzle.State, target: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return self.param_distance(self.params, current, target)

    def param_distance(self, params, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        x = self.pre_process(current, target)
        x = jnp.expand_dims(x, axis=0)
        x, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])
        return self.post_process(x)

    @abstractmethod
    def pre_process(self, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pass

    def post_process(self, x: chex.Array) -> float:
        """
        This function should return the post-processed distance.
        """
        return x.squeeze(1)

import pickle
from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from puzzle.puzzle_base import Puzzle

# Simba Residual Block
class ResBlock(nn.Module):
    stage1_features: int = 1000
    stage2_features: int = 1000

    @nn.compact
    def __call__(self, x0):
        x = nn.LayerNorm()(x0)
        x = nn.Dense(self.stage1_features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.stage2_features)(x)
        return x + x0


class DefaultModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = (x - 0.5) * 2.0 # normalize to [-1, 1]
        x = nn.Dense(1000)(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = ResBlock()(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x


class NeuralHeuristicBase(ABC):
    def __init__(self, puzzle: Puzzle, model: nn.Module = DefaultModel(), init_params: bool = True):
        self.puzzle = puzzle
        self.model = model
        dummy_current = self.puzzle.State.default()
        dummy_target = self.puzzle.State.default()
        if init_params:
            self.params = self.model.init(
                jax.random.PRNGKey(0), self.pre_process(dummy_current, dummy_target)
            )

    @classmethod
    def load_model(cls, puzzle: Puzzle, path: str):

        try:
            with open(path, "rb") as f:
                params = pickle.load(f)
            model = cls(puzzle, init_params=False)
            dummy_current = puzzle.State.default()
            dummy_target = puzzle.State.default()
            model.model.apply(
                params, model.pre_process(dummy_current, dummy_target)
            )  # check if the params are compatible with the model
            model.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            model = cls(puzzle)
        return model

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def distance(self, current: Puzzle.State, target: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return self.param_distance(self.params, current, target)
    
    def param_distance(self, params, current: Puzzle.State, target: Puzzle.State) -> chex.Array:
        x = self.pre_process(current, target)
        x = self.model.apply(params, x).squeeze()
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
        return x

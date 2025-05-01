import os
import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from neural_util.modules import DEFAULT_NORM_FN, DTYPE, ResBlock, conditional_dummy_norm
from neural_util.util import download_model, is_model_downloaded
from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


class QModelBase(nn.Module):
    action_size: int = 4
    Res_N: int = 4

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(5000, dtype=DTYPE)(x)
        x = DEFAULT_NORM_FN(x, training)
        x = nn.relu(x)
        x = nn.Dense(1000, dtype=DTYPE)(x)
        x = DEFAULT_NORM_FN(x, training)
        x = nn.relu(x)
        for _ in range(self.Res_N):
            x = ResBlock(1000)(x, training)
        x = nn.Dense(
            self.action_size, dtype=DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        _ = conditional_dummy_norm(x, training)
        return x


class NeuralQFunctionBase(QFunction):
    def __init__(
        self,
        puzzle: Puzzle,
        model: nn.Module = QModelBase,
        init_params: bool = True,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.is_fixed = puzzle.fixed_target
        self.action_size = self._get_action_size()
        self.model = model(self.action_size, **kwargs)
        if init_params:
            self.params = self.get_new_params()

    def _get_action_size(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return self.puzzle.get_neighbours(dummy_solve_config, dummy_current)[0].shape[0][0]

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

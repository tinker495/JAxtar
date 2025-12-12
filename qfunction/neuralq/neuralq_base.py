from abc import abstractmethod
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from puxle import Puzzle

from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    PreActivationResBlock,
    ResBlock,
    Swiglu,
)
from neural_util.nn_metadata import resolve_model_kwargs
from neural_util.param_manager import (
    load_params_with_metadata,
    save_params_with_metadata,
)
from neural_util.util import download_model, is_model_downloaded
from qfunction.q_base import QFunction


class QModelBase(nn.Module):
    action_size: int = 4
    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    activation: str = nn.relu
    norm_fn: callable = DEFAULT_NORM_FN
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False
    hidden_node_multiplier: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        if self.use_swiglu:
            x = Swiglu(self.initial_dim, norm_fn=self.norm_fn)(x)
            if self.resblock_fn != PreActivationResBlock:
                x = Swiglu(self.hidden_dim, norm_fn=self.norm_fn)(x)
            else:
                x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
        else:
            x = nn.Dense(self.initial_dim, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
            x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
            if self.resblock_fn != PreActivationResBlock:
                x = self.norm_fn(x, training)
                x = self.activation(x)
        for _ in range(self.Res_N):
            x = self.resblock_fn(
                self.hidden_dim * self.hidden_node_multiplier,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
            )(x, training)
        if self.resblock_fn == PreActivationResBlock:
            x = self.norm_fn(x, training)
            x = self.activation(x)
        x = x.astype(HEAD_DTYPE)
        x = nn.Dense(
            self.action_size, dtype=HEAD_DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        return x


class NeuralQFunctionBase(QFunction):
    def __init__(
        self,
        puzzle: Puzzle,
        model: nn.Module = QModelBase,
        init_params: bool = True,
        path: str = None,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.is_fixed = puzzle.fixed_target
        self.action_size = self._get_action_size()
        self.metadata = {}
        self.nn_args_metadata = {}
        self._preloaded_params = None
        self.path = path

        saved_metadata = {}
        if path is not None and not init_params:
            saved_metadata = self._preload_metadata()

        resolved_kwargs, nn_args = resolve_model_kwargs(kwargs, saved_metadata.get("nn_args"))
        self.nn_args_metadata = nn_args
        self.model = model(self.action_size, **resolved_kwargs)
        self.metadata = saved_metadata or {}
        self.metadata["nn_args"] = self.nn_args_metadata
        if path is not None:
            if init_params:
                self.params = self.get_new_params()
            else:
                self.params = self.load_model()
        else:
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

    def load_model(self):
        try:
            params = self._preloaded_params
            metadata = self.metadata
            if params is None:
                if not is_model_downloaded(self.path):
                    download_model(self.path)
                params, metadata = load_params_with_metadata(self.path)
            if params is None:
                print(
                    f"Warning: Loaded parameters from {self.path} are invalid or in an old format. "
                    "Initializing new parameters."
                )
                self.metadata = {}
                self.nn_args_metadata = {}
                return self.get_new_params()
            self.metadata = metadata or {}
            self.metadata["nn_args"] = self.nn_args_metadata

            dummy_solve_config = self.puzzle.SolveConfig.default()
            dummy_current = self.puzzle.State.default()
            self.model.apply(
                params,
                jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
                training=False,
            )  # check if the params are compatible with the model
            self._preloaded_params = None
            return params
        except Exception as e:
            print(f"Error loading model: {e}")
            return self.get_new_params()

    def save_model(self, path: str = None, metadata: dict = None):
        path = path or self.path
        if metadata is None:
            metadata = {}
        combined_metadata = {**self.metadata, **metadata}
        combined_metadata["puzzle_type"] = str(type(self.puzzle))
        combined_metadata["nn_args"] = self.nn_args_metadata
        save_params_with_metadata(path, self.params, combined_metadata)
        self.metadata = combined_metadata

    def prepare_q_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> tuple[Any, Puzzle.SolveConfig]:
        """
        This function prepares the parameters for use in Q-value calculations.
        By default, it returns the solve config unchanged. Subclasses can override
        this to transform the solve config into a more efficient representation,
        such as embedding it into a special vector (e.g., via a neural network)
        to guide the search toward the goal.
        """
        if "params" in kwargs and kwargs["params"] is not None:
            params = kwargs["params"]
        else:
            params = self.params
        return (params, solve_config)

    def batched_q_value(
        self, q_parameters: tuple[Any, Puzzle.SolveConfig], current: Puzzle.State
    ) -> chex.Array:
        params, solve_config = q_parameters
        return self.batched_param_q_value(params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.batched_pre_process(solve_config, current)
        x = self.model.apply(params, x, training=False)
        x = self.post_process(x)
        return x

    def batched_pre_process(
        self, solve_configs: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return jax.vmap(self.pre_process, in_axes=(None, 0))(solve_configs, current)

    def q_value(self, q_parameters: tuple[Any, Puzzle.SolveConfig], current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        params, solve_config = q_parameters
        return self.param_q_value(params, solve_config, current)[0]

    def param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        x = self.model.apply(params, x, training=False)
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

    def _preload_metadata(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            params, metadata = load_params_with_metadata(self.path)
            if params is not None:
                self._preloaded_params = params
            return metadata or {}
        except Exception as e:
            print(f"Error loading metadata from {self.path}: {e}")
            return {}

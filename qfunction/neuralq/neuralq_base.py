from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from puxle import Puzzle

from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    InputEncoder,
    ResBlock,
    conditional_dummy_norm,
    get_activation_fn,
    get_norm_fn,
)
from neural_util.param_manager import (
    load_params_with_metadata,
    save_params_with_metadata,
)
from neural_util.util import download_model, is_model_downloaded
from qfunction.q_base import QFunction


class QModelBase(nn.Module):
    action_size: int = 4
    Res_N: int = 4
    hidden_N: int = 1
    hidden_dim: int = 1000
    activation: str = nn.relu
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, x, training=False):
        x = InputEncoder(
            hidden_dim=self.hidden_dim,
            norm_fn=self.norm_fn,
            activation=self.activation,
        )(x, training)
        for _ in range(self.Res_N):
            x = ResBlock(
                self.hidden_dim,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
            )(x, training)
        x = nn.Dense(
            self.action_size, dtype=DTYPE, kernel_init=nn.initializers.normal(stddev=0.01)
        )(x)
        _ = conditional_dummy_norm(x, self.norm_fn, training)
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
        kwargs["norm_fn"] = get_norm_fn(kwargs.get("norm_fn", "batch"))
        kwargs["activation"] = get_activation_fn(kwargs.get("activation", "relu"))
        self.model = model(self.action_size, **kwargs)
        self.path = path
        self.metadata = {}
        if path is not None:
            if init_params:
                self.params = self.get_new_params()
            else:
                self.params = self.load_model()
        else:
            self.params = self.get_new_params()

    def _get_dummy_solve_config_and_current(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return dummy_solve_config, dummy_current

    def _get_dummy_pre_process(self):
        dummy_solve_config, dummy_current = self._get_dummy_solve_config_and_current()
        return jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0)

    def _get_action_size(self):
        dummy_solve_config, dummy_current = self._get_dummy_solve_config_and_current()
        return self.puzzle.get_neighbours(dummy_solve_config, dummy_current)[0].shape[0][0]

    def get_new_params(self):
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            self._get_dummy_pre_process(),
        )

    def load_model(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            params, metadata = load_params_with_metadata(self.path)
            if params is None:
                print(
                    f"Warning: Loaded parameters from {self.path} are invalid or in an old format. "
                    "Initializing new parameters."
                )
                self.metadata = {}
                return self.get_new_params()
            self.metadata = metadata

            self.model.apply(
                params,
                self._get_dummy_pre_process(),
                training=False,
            )  # check if the params are compatible with the model
            return params
        except Exception as e:
            print(f"Error loading model: {e}")
            return self.get_new_params()

    def save_model(self, path: str = None, metadata: dict = None):
        path = path or self.path
        if metadata is None:
            metadata = {}
        metadata["puzzle_type"] = str(type(self.puzzle))
        save_params_with_metadata(path, self.params, metadata)

    def batched_q_value(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_q_value(self.params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.batched_pre_process(solve_config, current)
        x, _ = self.model.apply(params, x, training=False, mutable=["batch_stats"])
        x = self.post_process(x)
        return x

    def batched_pre_process(
        self, solve_configs: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return jax.vmap(self.pre_process, in_axes=(None, 0))(solve_configs, current)

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

    def __str__(self):
        dummy_pre_process = self._get_dummy_pre_process()
        tabulate_fn = nn.tabulate(
            self.model, jax.random.key(0), compute_flops=True, compute_vjp_flops=True, depth=1
        )
        return tabulate_fn(dummy_pre_process)

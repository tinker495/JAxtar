from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from neural_util.modules import (
    DEFAULT_NORM_FN,
    DTYPE,
    HEAD_DTYPE,
    PreActivationResBlock,
    ResBlock,
    get_activation_fn,
    get_norm_fn,
    get_resblock_fn,
    swiglu_fn,
)
from neural_util.param_manager import (
    load_params_with_metadata,
    save_params_with_metadata,
)
from neural_util.util import download_model, is_model_downloaded


class HeuristicBase(nn.Module):

    Res_N: int = 4
    initial_dim: int = 5000
    hidden_N: int = 1
    hidden_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        if self.use_swiglu:
            x = swiglu_fn(self.initial_dim, self.activation, self.norm_fn)(x, training)
            x = swiglu_fn(self.hidden_dim, self.activation, self.norm_fn)(x, training)
        else:
            x = nn.Dense(self.initial_dim, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
            x = nn.Dense(self.hidden_dim, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
        for _ in range(self.Res_N):
            x = self.resblock_fn(
                self.hidden_dim,
                norm_fn=self.norm_fn,
                hidden_N=self.hidden_N,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
            )(x, training)
        if self.resblock_fn == PreActivationResBlock:
            x = self.activation(x)
            x = self.norm_fn(x, training)
        x = x.astype(HEAD_DTYPE)
        x = nn.Dense(1, dtype=HEAD_DTYPE, kernel_init=nn.initializers.normal(stddev=0.01))(x)
        return x


class NeuralHeuristicBase(Heuristic):
    def __init__(
        self,
        puzzle: Puzzle,
        model: nn.Module = HeuristicBase,
        init_params: bool = True,
        path: str = None,
        **kwargs,
    ):
        self.puzzle = puzzle
        kwargs["norm_fn"] = get_norm_fn(kwargs.get("norm_fn", "batch"))
        kwargs["activation"] = get_activation_fn(kwargs.get("activation", "relu"))
        kwargs["resblock_fn"] = get_resblock_fn(kwargs.get("resblock_fn", "standard"))
        kwargs["use_swiglu"] = kwargs.get("use_swiglu", False)
        self.model = model(**kwargs)
        self.is_fixed = puzzle.fixed_target
        self.path = path
        self.metadata = {}
        if path is not None:
            if init_params:
                self.params = self.get_new_params()
            else:
                self.params = self.load_model()
        else:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
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

            dummy_solve_config = self.puzzle.SolveConfig.default()
            dummy_current = self.puzzle.State.default()
            self.model.apply(
                params,
                jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
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

    def batched_distance(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_distance(self.params, solve_config, current)

    def batched_param_distance(
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

    def distance(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return float(self.param_distance(self.params, solve_config, current)[0])

    def param_distance(
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
        return x.squeeze(1)

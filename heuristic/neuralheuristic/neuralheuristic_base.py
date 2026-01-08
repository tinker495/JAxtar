import pickle
from abc import abstractmethod
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from heuristic.heuristic_base import Heuristic
from neural_util.aqt_utils import convert_to_serving
from neural_util.basemodel import DistanceHLGModel, DistanceModel, ResMLPModel
from neural_util.basemodel.selfpredictive import SelfPredictiveDistanceModel
from neural_util.nn_metadata import resolve_model_kwargs
from neural_util.param_manager import (
    load_params_with_metadata,
    merge_params,
    save_params_with_metadata,
)
from neural_util.util import download_model, is_model_downloaded


class NeuralHeuristicBase(Heuristic):
    def __init__(
        self,
        puzzle: Puzzle,
        model: DistanceModel | DistanceHLGModel = ResMLPModel,
        init_params: bool = True,
        path: str = None,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.is_fixed = puzzle.fixed_target
        self.action_size = puzzle.action_size
        self.path = path
        self.metadata = {}
        self.nn_args_metadata = {}
        self._preloaded_params = None

        saved_metadata = {}
        if self.path is not None and not init_params:
            saved_metadata = self._preload_metadata()

        resolved_kwargs, nn_args = resolve_model_kwargs(kwargs, saved_metadata.get("nn_args"))
        self.nn_args_metadata = nn_args
        self.aqt_cfg = kwargs.get("aqt_cfg")
        self.model_cls = model

        if issubclass(model, SelfPredictiveDistanceModel):
            self.model_kwargs = {**resolved_kwargs, "path_action_size": self.action_size}
            self.model = model(**resolved_kwargs, path_action_size=self.action_size)
        else:
            self.model_kwargs = resolved_kwargs
            self.model = model(**resolved_kwargs)

        self.metadata = saved_metadata or {}
        self.metadata["nn_args"] = self.nn_args_metadata

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

        # Check if the model has a specific initialization method (e.g., for self-predictive models)
        init_method = getattr(self.model, "initialize_components", None)

        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
            method=init_method,
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

            # Initialize full parameters for the current model configuration (includes AQT vars if any)
            full_params = self.get_new_params()

            # Merge old params into full params. This ensures AQT variables (if missing in old params) are present.
            params = merge_params(full_params, params)

            if self.aqt_cfg is not None:
                print("Converting model to serving mode (AQT)...")
                dummy_solve_config = self.puzzle.SolveConfig.default()
                dummy_current = self.puzzle.State.default()
                sample_input = jnp.expand_dims(
                    self.pre_process(dummy_solve_config, dummy_current), axis=0
                )

                self.model, params = convert_to_serving(
                    self.model_cls,
                    params,
                    sample_input,
                    **self.model_kwargs,
                )

            dummy_solve_config = self.puzzle.SolveConfig.default()
            dummy_current = self.puzzle.State.default()
            self.model.apply(
                params,
                jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0),
                training=False,
                rngs={"params": jax.random.PRNGKey(0)},
            )  # check if the params are compatible with the model
            self._preloaded_params = None
            return params
        except (
            FileNotFoundError,
            pickle.PickleError,
            ValueError,
            RuntimeError,
            OSError,
        ) as e:
            raise ValueError(f"Error loading NeuralHeuristic model: {e}") from e

    def save_model(self, path: str = None, metadata: dict = None):
        path = path or self.path
        if metadata is None:
            metadata = {}
        combined_metadata = {**self.metadata, **metadata}
        combined_metadata["puzzle_type"] = str(type(self.puzzle))
        combined_metadata["nn_args"] = self.nn_args_metadata
        save_params_with_metadata(path, self.params, combined_metadata)
        self.metadata = combined_metadata

    def prepare_heuristic_parameters(
        self, solve_config: Puzzle.SolveConfig, **kwargs: Any
    ) -> tuple[Any, Puzzle.SolveConfig]:
        """
        This function prepares the parameters for use in distance calculations.
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

    def batched_distance(
        self,
        heuristic_parameters: tuple[Any, Puzzle.SolveConfig],
        current: Puzzle.State,
    ) -> chex.Array:
        params, solve_config = heuristic_parameters
        return self.batched_param_distance(params, solve_config, current)

    def batched_param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.batched_pre_process(solve_config, current)
        x = self.model.apply(params, x, training=False, rngs={"params": jax.random.PRNGKey(0)})
        x = self.post_process(x)
        return x

    def batched_pre_process(
        self, solve_configs: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return jax.vmap(self.pre_process, in_axes=(None, 0))(solve_configs, current)

    def distance(
        self,
        heuristic_parameters: tuple[Any, Puzzle.SolveConfig],
        current: Puzzle.State,
    ) -> float:
        """
        This function should return the distance between the state and the target.
        """
        params, solve_config = heuristic_parameters
        return float(self.param_distance(params, solve_config, current)[0])

    def param_distance(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        x = self.model.apply(params, x, training=False, rngs={"params": jax.random.PRNGKey(0)})
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

    def _preload_metadata(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            params, metadata = load_params_with_metadata(self.path)
            if params is not None:
                self._preloaded_params = params
            return metadata or {}
        except (FileNotFoundError, pickle.PickleError, OSError, RuntimeError) as e:
            print(f"Error loading metadata from {self.path}: {e}")
            return {}

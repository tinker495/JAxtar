import pickle
from abc import ABC, abstractmethod
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle

from neural_util.aqt_utils import convert_to_serving
from neural_util.basemodel import DistanceHLGModel, DistanceModel, ResMLPModel
from neural_util.dtypes import INFERENCE_PARAM_DTYPE, PARAM_DTYPE
from neural_util.nn_metadata import resolve_model_kwargs
from neural_util.param_manager import (
    align_params_dtype,
    load_params_with_metadata,
    merge_params,
    save_params_with_metadata,
)
from neural_util.util import download_model, is_model_downloaded, resolve_model_path


class NeuralDistanceBase(ABC):
    load_error_name = "NeuralDistance"

    def __init__(
        self,
        puzzle: Puzzle,
        model: DistanceModel | DistanceHLGModel = ResMLPModel,
        init_params: bool = True,
        path: str = None,
        **kwargs,
    ):
        self.puzzle = puzzle
        self._configure_puzzle(puzzle)
        self.path = resolve_model_path(path) if path is not None else None
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
        self.model_kwargs = self._model_kwargs(resolved_kwargs)
        self.model = self._build_model(model, resolved_kwargs)

        self.metadata = saved_metadata or {}
        self.metadata["nn_args"] = self.nn_args_metadata
        if self.path is not None and not init_params:
            self.params = self.load_model()
        else:
            self.params = self.get_new_params()

    def _configure_puzzle(self, puzzle: Puzzle) -> None:
        pass

    def _model_kwargs(self, resolved_kwargs: dict[str, Any]) -> dict[str, Any]:
        return resolved_kwargs

    def _build_model(self, model, resolved_kwargs: dict[str, Any]):
        return model(**resolved_kwargs)

    def get_new_params(self):
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            self._sample_input(),
        )

    def load_model(self):
        try:
            params = self._preloaded_params
            metadata = self.metadata
            resolved_path = self.path
            if params is None:
                if resolved_path is None:
                    raise FileNotFoundError("Model path is required when init_params is False.")
                resolved_path = resolve_model_path(resolved_path)
                if not is_model_downloaded(resolved_path):
                    resolved_path = download_model(resolved_path)
                params, metadata = load_params_with_metadata(resolved_path)
            if resolved_path is not None:
                self.path = resolved_path
            assert params is not None, f"Failed to load parameters from {self.path}"
            self.metadata = metadata or {}
            self.metadata["nn_args"] = self.nn_args_metadata

            params = align_params_dtype(params, PARAM_DTYPE)
            params = merge_params(self.get_new_params(), params)

            if self.aqt_cfg is not None:
                print("Converting model to serving mode (AQT)...")
                self.model, params = convert_to_serving(
                    self.model_cls,
                    params,
                    self._sample_input(),
                    **self.model_kwargs,
                )
                if INFERENCE_PARAM_DTYPE is not None:
                    # Downcast everything except the frozen 'aqt' collection: its int8
                    # qvalues must stay int8 and its scale dtypes drive dequant dtypes.
                    params = {
                        col: tree
                        if col == "aqt"
                        else align_params_dtype(tree, INFERENCE_PARAM_DTYPE)
                        for col, tree in params.items()
                    }
            elif INFERENCE_PARAM_DTYPE is not None:
                params = align_params_dtype(params, INFERENCE_PARAM_DTYPE)

            self.model.apply(
                params,
                self._sample_input(),
                training=False,
                rngs={"params": jax.random.PRNGKey(0)},
            )
            self._preloaded_params = None
            return params
        except (
            FileNotFoundError,
            pickle.PickleError,
            ValueError,
            RuntimeError,
            OSError,
        ) as e:
            raise ValueError(f"Error loading {self.load_error_name} model: {e}") from e

    def save_model(self, path: str = None, metadata: dict = None):
        path = path or self.path
        if metadata is None:
            metadata = {}
        combined_metadata = {**self.metadata, **metadata}
        combined_metadata["puzzle_type"] = str(type(self.puzzle))
        combined_metadata["nn_args"] = self.nn_args_metadata
        save_params_with_metadata(path, self.params, combined_metadata)
        self.metadata = combined_metadata

    def _params_from_kwargs(self, **kwargs: Any):
        if "params" in kwargs and kwargs["params"] is not None:
            return kwargs["params"]
        return self.params

    def batched_pre_process(
        self, solve_configs: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return jax.vmap(self.pre_process, in_axes=(None, 0))(solve_configs, current)

    def _batched_model_values(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        x = self.batched_pre_process(solve_config, current)
        x = self.model.apply(params, x, training=False, rngs={"params": jax.random.PRNGKey(0)})
        return self.post_process(x)

    def _model_values(self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State):
        x = self.pre_process(solve_config, current)
        x = jnp.expand_dims(x, axis=0)
        x = self.model.apply(params, x, training=False, rngs={"params": jax.random.PRNGKey(0)})
        return self.post_process(x)

    def _sample_input(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return jnp.expand_dims(self.pre_process(dummy_solve_config, dummy_current), axis=0)

    @abstractmethod
    def pre_process(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> chex.Array:
        pass

    def post_process(self, x: chex.Array):
        return x

    def _preload_metadata(self):
        from neural_util.preprocessing import preload_metadata

        params, metadata, resolved_path = preload_metadata(
            self.path, is_model_downloaded, download_model, load_params_with_metadata
        )
        if params is not None:
            self._preloaded_params = params
        self.path = resolved_path
        return metadata

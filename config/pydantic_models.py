from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heuristic import EmptyHeuristic
from qfunction import EmptyQFunction


class PuzzleOptions(BaseModel):
    puzzle: str = "n-puzzle"
    puzzle_args: str = ""
    hard: bool = False
    seeds: str = "0"

    def get_seed_list(self) -> List[int]:
        if self.seeds.isdigit():
            return [int(self.seeds)]
        else:
            try:
                return [int(s) for s in self.seeds.split(",")]
            except ValueError:
                raise ValueError("Invalid seeds")


class SearchOptions(BaseModel):
    batch_size: int = Field(10000, description="Batch size for search.")
    max_node_size: int = Field(2000000, description="Maximum number of nodes to search.")
    cost_weight: float = Field(0.6, description="Weight for cost in search.")
    pop_ratio: float = Field(
        float("inf"),
        description=(
            "Controls the search beam width. Nodes are expanded if their cost is within `pop_ratio` "
            "percent of the best node's cost. For instance, 0.1 allows for a 10% margin. "
            "A value of 'inf' corresponds to a fixed-width beam search determined by the batch size."
        ),
    )
    vmap_size: int = Field(1, description="Size of vmap for search.")
    show_compile_time: bool = Field(False, description="Show compile time for search.")
    profile: bool = Field(False, description="Profile search.")
    debug: bool = Field(False, description="Debug mode.")

    def get_max_node_size(self):
        return int(self.max_node_size) // int(self.batch_size) * int(self.batch_size)


class EvalOptions(BaseModel):
    batch_size: Union[int, List[int]] = Field(
        [10000], description="Batch size for search. Can be a single int or a list of ints."
    )
    max_node_size: int = Field(int(2e7), description="Maximum number of nodes to search.")
    cost_weight: Union[float, List[float]] = Field(
        [0.9, 0.6, 0.3],
        description="Weight for cost in search. Can be a single float or a list of floats.",
    )
    pop_ratio: Union[float, List[float]] = Field(
        [float("inf"), 0.35, 0.2],
        description=(
            "Controls the search beam width. Nodes are expanded if their cost is within `pop_ratio` "
            "percent of the best node's cost. For instance, 0.1 allows for a 10% margin. "
            "A value of 'inf' corresponds to a fixed-width beam search determined by the batch size. "
            "Can be a single float or a list of floats for multiple evaluations."
        ),
    )
    num_eval: int = Field(200, description="Number of puzzles to evaluate.")
    run_name: Optional[str] = Field(None, description="Name of the evaluation run.")
    scatter_max_points: int = Field(
        200, description="Maximum number of points to plot in scatter plots."
    )
    max_expansion_plots: int = Field(
        3,
        description="Maximum number of individual expansion plots to generate per run. Set to 0 to disable.",
    )
    use_early_stopping: bool = Field(
        True, description="Enable early stopping based on success rate threshold."
    )
    early_stop_patience: int = Field(
        10, description="Number of samples to check before considering early stopping."
    )
    early_stop_threshold: float = Field(
        0.5, description="Minimum success rate threshold for early stopping (0.0 to 1.0)."
    )

    def get_max_node_size(self, batch_size: int) -> int:
        return self.max_node_size // batch_size * batch_size

    def light_eval(self, max_eval: int = 20) -> "EvalOptions":
        capped_eval = min(max_eval, self.num_eval)
        return self.model_copy(
            update={
                "num_eval": capped_eval,
                "cost_weight": [self.cost_weight[0]],
                "pop_ratio": [self.pop_ratio[0]],
            }
        )

    @property
    def light_eval_options(self) -> "EvalOptions":
        return self.light_eval()


class VisualizeOptions(BaseModel):
    visualize_terminal: bool = Field(False, description="Visualize path in terminal.")
    visualize_imgs: bool = False
    max_animation_time: int = 10


class HeuristicOptions(BaseModel):
    neural_heuristic: bool = False
    param_path: Optional[str] = None


class QFunctionOptions(BaseModel):
    neural_qfunction: bool = False
    param_path: Optional[str] = None


class DistTrainOptions(BaseModel):
    steps: int = int(5e3)
    dataset_batch_size: int = 8192 * 256
    dataset_minibatch_size: int = 8192 * 16
    train_minibatch_size: int = 8192
    key: int = 0
    reset: bool = True
    loss_threshold: float = float("inf")
    update_interval: int = 32
    force_update_interval: int = 2048
    use_soft_update: bool = False
    use_double_dqn: bool = False
    using_hindsight_target: bool = False
    using_triangular_sampling: bool = False
    use_diffusion_distance: bool = False
    debug: bool = False
    multi_device: bool = True
    reset_interval: int = int(1e6) # just large enough
    tau: float = 0.2
    learning_rate: float = 1e-3
    weight_decay_size: Optional[float] = 0.0
    opt_state_reset: bool = False
    optimizer: str = "adam"
    temperature: float = 0.33
    replay_ratio: int = Field(
        1, description="Number of gradient updates per generated dataset. Default is 1."
    )
    logger: str = Field("aim", description="Logger to use. Can be 'aim', 'tensorboard', or 'none'.")
    loss: str = Field(
        "mse",
        description=(
            "Training loss: 'mse', 'huber', 'logcosh', 'asymmetric_huber', or 'asymmetric_logcosh'."
        ),
    )
    loss_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the selected loss (JSON key/value).",
    )
    td_error_clip: Optional[float] = Field(
        None, description="Absolute clip value applied to TD-error."
    )


class WMDatasetOptions(BaseModel):
    dataset_size: int = 300000
    dataset_minibatch_size: int = 30000
    shuffle_length: int = 30
    img_size: Tuple[int, int] = (32, 32)
    key: int = 0


class WMGetDSOptions(BaseModel):
    dataset: str = "rubikscube"


class WMGetModelOptions(BaseModel):
    world_model: str = "rubikscube"


class WMTrainOptions(BaseModel):
    train_epochs: int = 2000
    mini_batch_size: int = 1000
    optimizer: str = "adam"


class NeuralCallableConfig(BaseModel):
    callable: Callable
    path_template: str
    neural_config: Optional[dict] = {}

    class Config:
        arbitrary_types_allowed = True


class WorldModelPuzzleConfig(BaseModel):
    callable: Callable
    path: str
    neural_config: Optional[dict] = {}

    class Config:
        arbitrary_types_allowed = True


class PuzzleConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    callable: Callable
    initial_shuffle: Optional[int] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _merge_extras(cls, values):
        if isinstance(values, cls) or not isinstance(values, dict):
            return values

        data = dict(values)
        recognized = {"callable", "initial_shuffle", "kwargs"}
        extra = {k: data.pop(k) for k in list(data.keys()) if k not in recognized}

        if extra:
            merged_kwargs = dict(data.get("kwargs", {}))
            merged_kwargs.update(extra)
            data["kwargs"] = merged_kwargs

        return data


class PuzzleBundle(BaseModel):
    puzzle: Optional[Union[Callable, PuzzleConfig, WorldModelPuzzleConfig]] = None
    puzzle_hard: Optional[Union[Callable, PuzzleConfig]] = None
    puzzle_ds: Optional[Callable] = None
    heuristic: Callable = EmptyHeuristic
    heuristic_nn_config: Optional[NeuralCallableConfig] = None
    q_function: Callable = EmptyQFunction
    q_function_nn_config: Optional[NeuralCallableConfig] = None
    k_max: int = 50
    eval_options: EvalOptions = Field(default_factory=EvalOptions)
    search_options: SearchOptions = Field(default_factory=SearchOptions)

    class Config:
        arbitrary_types_allowed = True


class WorldModelBundle(BaseModel):
    world_model: Callable
    dataset_path: str
    puzzle_for_ds_gen: Callable

    class Config:
        arbitrary_types_allowed = True

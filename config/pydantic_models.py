from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field

from heuristic import EmptyHeuristic
from qfunction import EmptyQFunction


class PuzzleOptions(BaseModel):
    puzzle: str = "n-puzzle"
    puzzle_size: str = "default"
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
    cost_weight: float = Field(0.5, description="Weight for cost in search.")
    vmap_size: int = Field(1, description="Size of vmap for search.")
    show_compile_time: bool = Field(False, description="Show compile time for search.")
    profile: bool = Field(False, description="Profile search.")

    def get_max_node_size(self):
        return self.max_node_size // self.batch_size * self.batch_size


class EvalOptions(BaseModel):
    batch_size: int = Field(10000, description="Batch size for search.")
    max_node_size: int = Field(int(2e7), description="Maximum number of nodes to search.")
    cost_weight: float = Field(0.6, description="Weight for cost in search.")
    num_eval: int = Field(200, description="Number of puzzles to evaluate.")
    run_name: Optional[str] = Field(None, description="Name of the evaluation run.")

    def get_max_node_size(self):
        return self.max_node_size // self.batch_size * self.batch_size


class VisualizeOptions(BaseModel):
    visualize_terminal: bool = Field(False, description="Visualize path in terminal.")
    visualize_imgs: bool = False
    max_animation_time: int = 10


class HeuristicOptions(BaseModel):
    neural_heuristic: bool = False


class QFunctionOptions(BaseModel):
    neural_qfunction: bool = False


class DistTrainOptions(BaseModel):
    steps: int = int(2e4)
    dataset_batch_size: int = 524288
    dataset_minibatch_size: int = 8192
    train_minibatch_size: int = 8192
    key: int = 0
    reset: bool = True
    loss_threshold: float = float("inf")
    update_interval: int = 128
    use_soft_update: bool = False
    using_hindsight_target: bool = False
    using_importance_sampling: bool = False
    using_triangular_sampling: bool = False
    use_target_confidence_weighting: bool = False
    debug: bool = False
    multi_device: bool = True
    reset_interval: int = 4000
    tau: float = 0.2
    opt_state_reset: bool = True
    optimizer: str = "adam"


class DistQFunctionOptions(BaseModel):
    not_with_policy: bool = False


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


class PuzzleBundle(BaseModel):
    puzzle: Optional[Callable] = None
    puzzle_hard: Optional[Callable] = None
    puzzle_ds: Optional[Callable] = None
    heuristic: Callable = EmptyHeuristic
    heuristic_nn: Optional[Callable] = None
    q_function: Callable = EmptyQFunction
    q_function_nn: Optional[Callable] = None
    shuffle_length: int = 50

    class Config:
        arbitrary_types_allowed = True


class WorldModelBundle(BaseModel):
    world_model: Callable
    dataset_path: str
    puzzle_for_ds_gen: Callable

    class Config:
        arbitrary_types_allowed = True

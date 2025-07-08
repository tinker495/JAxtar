from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel

from helpers.formatting import human_format_to_float
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
    max_node_size: str = "2e6"
    batch_size: int = int(1e4)
    cost_weight: float = 0.9
    vmap_size: int = 1
    debug: bool = False
    profile: bool = False
    show_compile_time: bool = False

    def get_max_node_size(self) -> int:
        return int(human_format_to_float(self.max_node_size))


class VisualizeOptions(BaseModel):
    visualize_terminal: bool = False
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
    debug: bool = False
    multi_device: bool = True
    reset_interval: int = 4000
    tau: float = 0.2


class DistQFunctionOptions(BaseModel):
    not_with_policy: bool = False


class WBSDistTrainOptions(BaseModel):
    steps: int = int(2e3)  # 50 * 2e4 = 1e6 / DeepCubeA settings
    replay_size: int = int(1e8)
    max_nodes: int = int(2e7)
    add_batch_size: int = 524288  # 8192 * 64
    search_batch_size: int = 8192  # 8192 * 64
    train_minibatch_size: int = 8192  # 128 * 16
    sample_ratio: float = 0.3
    cost_weight: float = 0.8
    key: int = 0
    reset: bool = False
    use_optimal_branch: bool = False
    debug: bool = False
    multi_device: bool = False


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

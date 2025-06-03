from puzzle import Puzzle, RubiksCubeDS, SokobanDS
from puzzle.world_model import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModelOptimized,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
)

puzzle_dict_ds: dict[str, Puzzle] = {
    "rubikscube": RubiksCubeDS,
    "sokoban": SokobanDS,
}

world_model_dict: dict[str, callable] = {
    "rubikscube": lambda reset: RubiksCubeWorldModel(
        init_params=reset, path="puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_optimized": lambda reset: RubiksCubeWorldModelOptimized(
        init_params=reset, path="puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "sokoban": lambda reset: SokobanWorldModel(
        init_params=reset, path="puzzle/world_model/model/params/sokoban.pkl"
    ),
    "sokoban_optimized": lambda reset: SokobanWorldModelOptimized(
        init_params=reset, path="puzzle/world_model/model/params/sokoban_optimized.pkl"
    ),
}


world_model_ds_dict: dict[str, str] = {
    "rubikscube": "puzzle/world_model/data/rubikscube/transition",
    "sokoban": "puzzle/world_model/data/sokoban/transition",
}

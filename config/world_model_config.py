from puxle import Puzzle, RubiksCube, Sokoban

from world_model_puzzle import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModelOptimized,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
)

puzzle_dict_ds: dict[str, Puzzle] = {
    "rubikscube": lambda **kwargs: RubiksCube(initial_shuffle=1000, **kwargs),
    "sokoban": lambda **kwargs: Sokoban(
        solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET, **kwargs
    ),
}

world_model_dict: dict[str, callable] = {
    "rubikscube": lambda reset: RubiksCubeWorldModel(
        init_params=reset, path="world_model_puzzle/model/params/rubikscube.pkl"
    ),
    "rubikscube_optimized": lambda reset: RubiksCubeWorldModelOptimized(
        init_params=reset, path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
    ),
    "sokoban": lambda reset: SokobanWorldModel(
        init_params=reset, path="world_model_puzzle/model/params/sokoban.pkl"
    ),
    "sokoban_optimized": lambda reset: SokobanWorldModelOptimized(
        init_params=reset, path="world_model_puzzle/model/params/sokoban_optimized.pkl"
    ),
}


world_model_ds_dict: dict[str, str] = {
    "rubikscube": "world_model_puzzle/data/rubikscube/transition",
    "sokoban": "world_model_puzzle/data/sokoban/transition",
}

from typing import Dict

from puxle import RubiksCube, Sokoban

from world_model_puzzle import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModelOptimized,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
)

from .pydantic_models import WorldModelBundle

world_model_bundles: Dict[str, WorldModelBundle] = {
    "rubikscube": WorldModelBundle(
        puzzle_for_ds_gen=lambda **kwargs: RubiksCube(initial_shuffle=1000, **kwargs),
        world_model=lambda reset: RubiksCubeWorldModel(
            init_params=reset, path="world_model_puzzle/model/params/rubikscube.pkl"
        ),
        dataset_path="world_model_puzzle/data/rubikscube/transition",
    ),
    "rubikscube_optimized": WorldModelBundle(
        puzzle_for_ds_gen=lambda **kwargs: RubiksCube(initial_shuffle=1000, **kwargs),
        world_model=lambda reset: RubiksCubeWorldModelOptimized(
            init_params=reset,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        dataset_path="world_model_puzzle/data/rubikscube/transition",
    ),
    "sokoban": WorldModelBundle(
        puzzle_for_ds_gen=lambda **kwargs: Sokoban(
            solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET,
            **kwargs,
        ),
        world_model=lambda reset: SokobanWorldModel(
            init_params=reset, path="world_model_puzzle/model/params/sokoban.pkl"
        ),
        dataset_path="world_model_puzzle/data/sokoban/transition",
    ),
    "sokoban_optimized": WorldModelBundle(
        puzzle_for_ds_gen=lambda **kwargs: Sokoban(
            solve_condition=Sokoban.SolveCondition.ALL_BOXES_ON_TARGET_AND_PLAYER_ON_TARGET,
            **kwargs,
        ),
        world_model=lambda reset: SokobanWorldModelOptimized(
            init_params=reset, path="world_model_puzzle/model/params/sokoban_optimized.pkl"
        ),
        dataset_path="world_model_puzzle/data/sokoban/transition",
    ),
}

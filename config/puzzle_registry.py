from typing import Dict

from puxle import (
    TSP,
    DotKnot,
    LightsOut,
    Maze,
    PancakeSorting,
    Room,
    RubiksCube,
    RubiksCubeRandom,
    SlidePuzzle,
    SlidePuzzleHard,
    SlidePuzzleRandom,
    Sokoban,
    SokobanHard,
    TopSpin,
    TowerOfHanoi,
)

from heuristic import (
    DotKnotHeuristic,
    LightsOutHeuristic,
    MazeHeuristic,
    PancakeHeuristic,
    RubiksCubeHeuristic,
    SlidePuzzleHeuristic,
    SokobanHeuristic,
    TSPHeuristic,
)
from heuristic.neuralheuristic import (
    LightsOutConvNeuralHeuristic,
    LightsOutNeuralHeuristic,
    PancakeNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    SlidePuzzleConvNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
    SokobanNeuralHeuristic,
    WorldModelNeuralHeuristic,
)
from qfunction import (
    TSPQ,
    DotKnotQ,
    LightsOutQ,
    MazeQ,
    PancakeQ,
    RubiksCubeQ,
    SlidePuzzleQ,
    SokobanQ,
)
from qfunction.neuralq import (
    LightsOutConvNeuralQ,
    LightsOutNeuralQ,
    PancakeNeuralQ,
    RubiksCubeNeuralQ,
    SlidePuzzleConvNeuralQ,
    SlidePuzzleNeuralQ,
    SokobanNeuralQ,
    WorldModelNeuralQ,
)
from world_model_puzzle import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModel_reversed,
    RubiksCubeWorldModel_test,
    RubiksCubeWorldModelOptimized,
    RubiksCubeWorldModelOptimized_reversed,
    RubiksCubeWorldModelOptimized_test,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
)

from .pydantic_models import EvalOptions, PuzzleBundle, SearchOptions

puzzle_bundles: Dict[str, PuzzleBundle] = {
    "n-puzzle": PuzzleBundle(
        puzzle=SlidePuzzle,
        puzzle_hard=SlidePuzzleHard,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn=lambda puzzle, reset: SlidePuzzleNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/n-puzzle_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: SlidePuzzleNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/n-puzzle_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "n-puzzle-conv": PuzzleBundle(
        puzzle=SlidePuzzle,
        puzzle_hard=SlidePuzzleHard,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn=lambda puzzle, reset: SlidePuzzleConvNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/n-puzzle-conv_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: SlidePuzzleConvNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/n-puzzle-conv_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "n-puzzle-random": PuzzleBundle(
        puzzle=SlidePuzzleRandom,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn=lambda puzzle, reset: SlidePuzzleNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/n-puzzle-random_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: SlidePuzzleNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/n-puzzle-random_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "lightsout": PuzzleBundle(
        puzzle=LightsOut,
        puzzle_hard=lambda **kwargs: LightsOut(initial_shuffle=50, **kwargs),
        heuristic=LightsOutHeuristic,
        q_function=LightsOutQ,
        heuristic_nn=lambda puzzle, reset: LightsOutNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/lightsout_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: LightsOutNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/lightsout_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "lightsout-conv": PuzzleBundle(
        puzzle=LightsOut,
        puzzle_hard=lambda **kwargs: LightsOut(initial_shuffle=50, **kwargs),
        heuristic=LightsOutHeuristic,
        q_function=LightsOutQ,
        heuristic_nn=lambda puzzle, reset: LightsOutConvNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/lightsout-conv_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: LightsOutConvNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/lightsout-conv_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube": PuzzleBundle(
        puzzle=RubiksCube,
        puzzle_hard=lambda **kwargs: RubiksCube(initial_shuffle=50, **kwargs),
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn=lambda puzzle, reset: RubiksCubeNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/rubikscube_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: RubiksCubeNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/rubikscube_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube-random": PuzzleBundle(
        puzzle=RubiksCubeRandom,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn=lambda puzzle, reset: RubiksCubeNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/rubikscube-random_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: RubiksCubeNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/rubikscube-random_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "maze": PuzzleBundle(puzzle=Maze, heuristic=MazeHeuristic, q_function=MazeQ),
    "room": PuzzleBundle(puzzle=Room, heuristic=MazeHeuristic, q_function=MazeQ),
    "dotknot": PuzzleBundle(puzzle=DotKnot, heuristic=DotKnotHeuristic, q_function=DotKnotQ),
    "tsp": PuzzleBundle(puzzle=TSP, heuristic=TSPHeuristic, q_function=TSPQ),
    "sokoban": PuzzleBundle(
        puzzle=Sokoban,
        puzzle_hard=SokobanHard,
        heuristic=SokobanHeuristic,
        q_function=SokobanQ,
        heuristic_nn=lambda puzzle, reset: SokobanNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/sokoban_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: SokobanNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/sokoban_{puzzle.size}.pkl",
            init_params=reset,
        ),
        eval_options=EvalOptions(
            batch_size=100,
        ),
        search_options=SearchOptions(
            batch_size=100,
        ),
    ),
    "pancake": PuzzleBundle(
        puzzle=PancakeSorting,
        heuristic=PancakeHeuristic,
        q_function=PancakeQ,
        heuristic_nn=lambda puzzle, reset: PancakeNeuralHeuristic(
            puzzle=puzzle,
            path=f"heuristic/neuralheuristic/model/params/pancake_{puzzle.size}.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: PancakeNeuralQ(
            puzzle=puzzle,
            path=f"qfunction/neuralq/model/params/pancake_{puzzle.size}.pkl",
            init_params=reset,
        ),
    ),
    "hanoi": PuzzleBundle(puzzle=TowerOfHanoi),
    "topspin": PuzzleBundle(puzzle=TopSpin),
    "rubikscube_world_model": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModel(
            path="world_model_puzzle/model/params/rubikscube.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube_world_model_test": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModel_test(
            path="world_model_puzzle/model/params/rubikscube.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube_world_model_reversed": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModel_reversed(
            path="world_model_puzzle/model/params/rubikscube.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube_world_model_optimized": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModelOptimized(
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube_world_model_optimized_test": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModelOptimized_test(
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
    ),
    "rubikscube_world_model_optimized_reversed": PuzzleBundle(
        puzzle=lambda **kwargs: RubiksCubeWorldModelOptimized_reversed(
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            init_params=reset,
        ),
    ),
    "sokoban_world_model": PuzzleBundle(
        puzzle=lambda **kwargs: SokobanWorldModel(
            path="world_model_puzzle/model/params/sokoban.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/sokoban_world_model_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/sokoban_world_model_None.pkl",
            init_params=reset,
        ),
    ),
    "sokoban_world_model_optimized": PuzzleBundle(
        puzzle=lambda **kwargs: SokobanWorldModelOptimized(
            path="world_model_puzzle/model/params/sokoban_optimized.pkl"
        ),
        heuristic_nn=lambda puzzle, reset: WorldModelNeuralHeuristic(
            puzzle=puzzle,
            path="heuristic/neuralheuristic/model/params/sokoban_world_model_optimized_None.pkl",
            init_params=reset,
        ),
        q_function_nn=lambda puzzle, reset: WorldModelNeuralQ(
            puzzle=puzzle,
            path="qfunction/neuralq/model/params/sokoban_world_model_optimized_None.pkl",
            init_params=reset,
        ),
        eval_options=EvalOptions(
            batch_size=100,
        ),
        search_options=SearchOptions(
            batch_size=100,
        ),
    ),
}

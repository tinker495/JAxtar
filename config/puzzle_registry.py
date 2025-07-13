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

from .pydantic_models import (
    EvalOptions,
    NeuralCallableConfig,
    PuzzleBundle,
    PuzzleConfig,
    SearchOptions,
    WorldModelPuzzleConfig,
)

puzzle_bundles: Dict[str, PuzzleBundle] = {
    "n-puzzle": PuzzleBundle(
        puzzle=SlidePuzzle,
        puzzle_hard=SlidePuzzleHard,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/n-puzzle_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            path_template="qfunction/neuralq/model/params/n-puzzle_{size}.pkl",
        ),
    ),
    "n-puzzle-conv": PuzzleBundle(
        puzzle=SlidePuzzle,
        puzzle_hard=SlidePuzzleHard,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleConvNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/n-puzzle-conv_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleConvNeuralQ,
            path_template="qfunction/neuralq/model/params/n-puzzle-conv_{size}.pkl",
        ),
    ),
    "n-puzzle-random": PuzzleBundle(
        puzzle=SlidePuzzleRandom,
        shuffle_length=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/n-puzzle-random_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            path_template="qfunction/neuralq/model/params/n-puzzle-random_{size}.pkl",
        ),
    ),
    "lightsout": PuzzleBundle(
        puzzle=LightsOut,
        puzzle_hard=PuzzleConfig(callable=LightsOut, initial_shuffle=50),
        heuristic=LightsOutHeuristic,
        q_function=LightsOutQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=LightsOutNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/lightsout_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=LightsOutNeuralQ,
            path_template="qfunction/neuralq/model/params/lightsout_{size}.pkl",
        ),
    ),
    "lightsout-conv": PuzzleBundle(
        puzzle=LightsOut,
        puzzle_hard=PuzzleConfig(callable=LightsOut, initial_shuffle=50),
        heuristic=LightsOutHeuristic,
        q_function=LightsOutQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=LightsOutConvNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/lightsout-conv_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=LightsOutConvNeuralQ,
            path_template="qfunction/neuralq/model/params/lightsout-conv_{size}.pkl",
        ),
    ),
    "rubikscube": PuzzleBundle(
        puzzle=RubiksCube,
        puzzle_hard=PuzzleConfig(callable=RubiksCube, initial_shuffle=50),
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_{size}.pkl",
        ),
    ),
    "rubikscube-random": PuzzleBundle(
        puzzle=RubiksCubeRandom,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_{size}.pkl",
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
        heuristic_nn_config=NeuralCallableConfig(
            callable=SokobanNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/sokoban_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SokobanNeuralQ,
            path_template="qfunction/neuralq/model/params/sokoban_{size}.pkl",
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
        heuristic_nn_config=NeuralCallableConfig(
            callable=PancakeNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/pancake_{size}.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=PancakeNeuralQ,
            path_template="qfunction/neuralq/model/params/pancake_{size}.pkl",
        ),
    ),
    "hanoi": PuzzleBundle(puzzle=TowerOfHanoi),
    "topspin": PuzzleBundle(puzzle=TopSpin),
    "rubikscube_world_model": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel, path="world_model_puzzle/model/params/rubikscube.pkl"
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        ),
    ),
    "rubikscube_world_model_test": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel_test,
            path="world_model_puzzle/model/params/rubikscube.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        ),
    ),
    "rubikscube_world_model_reversed": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel_reversed,
            path="world_model_puzzle/model/params/rubikscube.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        ),
    ),
    "rubikscube_world_model_optimized": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
    ),
    "rubikscube_world_model_optimized_test": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized_test,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
    ),
    "rubikscube_world_model_optimized_reversed": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized_reversed,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        ),
    ),
    "sokoban_world_model": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=SokobanWorldModel, path="world_model_puzzle/model/params/sokoban.pkl"
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/sokoban_world_model_optimized_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/sokoban_world_model_optimized_None.pkl",
        ),
    ),
    "sokoban_world_model_optimized": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=SokobanWorldModelOptimized,
            path="world_model_puzzle/model/params/sokoban_optimized.pkl",
        ),
        heuristic_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralHeuristic,
            path_template="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=WorldModelNeuralQ,
            path_template="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        ),
        eval_options=EvalOptions(
            batch_size=100,
        ),
        search_options=SearchOptions(
            batch_size=100,
        ),
    ),
}

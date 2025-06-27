from puxle import (
    TSP,
    DotKnot,
    LightsOut,
    Maze,
    PancakeSorting,
    Puzzle,
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
    EmptyHeuristic,
    Heuristic,
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
    EmptyQFunction,
    LightsOutQ,
    MazeQ,
    PancakeQ,
    QFunction,
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
    RubiksCubeWorldModelOptimized,
    RubiksCubeWorldModelOptimizedReversed,
    RubiksCubeWorldModelOptimizedTest,
    RubiksCubeWorldModelReversed,
    RubiksCubeWorldModelTest,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
)

default_puzzle_sizes: dict[str, int] = {
    "n-puzzle": 4,
    "n-puzzle-conv": 4,
    "n-puzzle-random": 4,
    "n-puzzle-conv-random": 4,
    "lightsout": 7,
    "lightsout-conv": 7,
    "rubikscube": 3,
    "rubikscube-random": 3,
    "maze": 23,
    "tsp": 16,
    "dotknot": 8,
    "sokoban": 10,
    "pancake": 35,
    "hanoi": 10,
    "topspin": 20,
    "rubikscube_world_model": None,
    "rubikscube_world_model_test": None,
    "rubikscube_world_model_reversed": None,
    "rubikscube_world_model_optimized": None,
    "rubikscube_world_model_optimized_test": None,
    "rubikscube_world_model_optimized_reversed": None,
    "sokoban_world_model": None,
    "sokoban_world_model_optimized": None,
}

puzzle_dict: dict[str, Puzzle] = {
    "n-puzzle": SlidePuzzle,
    "n-puzzle-conv": SlidePuzzle,
    "n-puzzle-random": SlidePuzzleRandom,
    "n-puzzle-conv-random": SlidePuzzleRandom,
    "lightsout": LightsOut,
    "lightsout-conv": LightsOut,
    "rubikscube": RubiksCube,
    "rubikscube-random": RubiksCubeRandom,
    "maze": Maze,
    "dotknot": DotKnot,
    "tsp": TSP,
    "sokoban": Sokoban,
    "pancake": PancakeSorting,
    "hanoi": TowerOfHanoi,
    "topspin": TopSpin,
    "rubikscube_world_model": lambda **kwargs: RubiksCubeWorldModel(
        path="world_model_puzzle/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_test": lambda **kwargs: RubiksCubeWorldModelTest(
        path="world_model_puzzle/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_reversed": lambda **kwargs: RubiksCubeWorldModelReversed(
        path="world_model_puzzle/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_optimized": lambda **kwargs: RubiksCubeWorldModelOptimized(
        path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_test": lambda **kwargs: RubiksCubeWorldModelOptimizedTest(
        path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_reversed": lambda **kwargs: RubiksCubeWorldModelOptimizedReversed(
        path="world_model_puzzle/model/params/rubikscube_optimized.pkl"
    ),
    "sokoban_world_model": lambda **kwargs: SokobanWorldModel(
        path="world_model_puzzle/model/params/sokoban.pkl"
    ),
    "sokoban_world_model_optimized": lambda **kwargs: SokobanWorldModelOptimized(
        path="world_model_puzzle/model/params/sokoban_optimized.pkl"
    ),
}

puzzle_dict_hard: dict[str, Puzzle] = {
    "n-puzzle": SlidePuzzleHard,
    "n-puzzle-conv": SlidePuzzleHard,
    "lightsout": lambda **kwargs: LightsOut(initial_shuffle=50, **kwargs),
    "lightsout-conv": lambda **kwargs: LightsOut(initial_shuffle=50, **kwargs),
    "rubikscube": lambda **kwargs: RubiksCube(initial_shuffle=50, **kwargs),
    "sokoban": SokobanHard,
}

puzzle_heuristic_dict: dict[str, Heuristic] = {
    "n-puzzle": SlidePuzzleHeuristic,
    "n-puzzle-conv": SlidePuzzleHeuristic,
    "n-puzzle-random": SlidePuzzleHeuristic,
    "n-puzzle-conv-random": SlidePuzzleHeuristic,
    "lightsout": LightsOutHeuristic,
    "lightsout-conv": LightsOutHeuristic,
    "rubikscube": RubiksCubeHeuristic,
    "rubikscube-random": RubiksCubeHeuristic,
    "maze": MazeHeuristic,
    "dotknot": DotKnotHeuristic,
    "tsp": TSPHeuristic,
    "sokoban": SokobanHeuristic,
    "pancake": PancakeHeuristic,
    "hanoi": EmptyHeuristic,
    "topspin": EmptyHeuristic,
    "rubikscube_world_model": EmptyHeuristic,
    "rubikscube_world_model_test": EmptyHeuristic,
    "rubikscube_world_model_optimized": EmptyHeuristic,
    "rubikscube_world_model_optimized_test": EmptyHeuristic,
    "sokoban_world_model": EmptyHeuristic,
    "sokoban_world_model_optimized": EmptyHeuristic,
}

# nn option need to be callable, for loading model
puzzle_heuristic_dict_nn: dict[str, callable] = {
    "n-puzzle": lambda puzzle, reset: SlidePuzzleNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "n-puzzle-random": lambda puzzle, reset: SlidePuzzleNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle-random_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "n-puzzle-conv": lambda puzzle, reset: SlidePuzzleConvNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle-conv_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "lightsout": lambda puzzle, reset: LightsOutNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/lightsout_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "lightsout-conv": lambda puzzle, reset: LightsOutConvNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/lightsout-conv_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube": lambda puzzle, reset: RubiksCubeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/rubikscube_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube-random": lambda puzzle, reset: RubiksCubeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/rubikscube-random_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "sokoban": lambda puzzle, reset: SokobanNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/sokoban_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "pancake": lambda puzzle, reset: PancakeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/pancake_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_test": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_reversed": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_test": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_reversed": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/sokoban_world_model_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model_optimized": lambda puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/sokoban_world_model_optimized_None.pkl",
        init_params=reset,
    ),
}

puzzle_q_function_dict: dict[str, QFunction] = {
    "n-puzzle": SlidePuzzleQ,
    "n-puzzle-conv": SlidePuzzleQ,
    "n-puzzle-random": SlidePuzzleQ,
    "n-puzzle-conv-random": SlidePuzzleQ,
    "lightsout": LightsOutQ,
    "lightsout-conv": LightsOutQ,
    "rubikscube": RubiksCubeQ,
    "rubikscube-random": RubiksCubeQ,
    "maze": MazeQ,
    "dotknot": DotKnotQ,
    "tsp": TSPQ,
    "sokoban": SokobanQ,
    "pancake": PancakeQ,
    "hanoi": EmptyQFunction,
    "topspin": EmptyQFunction,
    "rubikscube_world_model": EmptyQFunction,
    "rubikscube_world_model_test": EmptyQFunction,
    "rubikscube_world_model_reversed": EmptyQFunction,
    "rubikscube_world_model_optimized": EmptyQFunction,
    "rubikscube_world_model_optimized_test": EmptyQFunction,
    "rubikscube_world_model_optimized_reversed": EmptyQFunction,
    "sokoban_world_model": EmptyQFunction,
    "sokoban_world_model_optimized": EmptyQFunction,
}

puzzle_q_function_dict_nn: dict[str, callable] = {
    "n-puzzle": lambda puzzle, reset: SlidePuzzleNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "n-puzzle-random": lambda puzzle, reset: SlidePuzzleNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle-random_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "n-puzzle-conv": lambda puzzle, reset: SlidePuzzleConvNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle-conv_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "lightsout": lambda puzzle, reset: LightsOutNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/lightsout_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "lightsout-conv": lambda puzzle, reset: LightsOutConvNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/lightsout-conv_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube": lambda puzzle, reset: RubiksCubeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/rubikscube_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube-random": lambda puzzle, reset: RubiksCubeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/rubikscube-random_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "sokoban": lambda puzzle, reset: SokobanNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/sokoban_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "pancake": lambda puzzle, reset: PancakeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/pancake_{puzzle.size}.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_test": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_reversed": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_test": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_reversed": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/sokoban_world_model_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model_optimized": lambda puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/sokoban_world_model_optimized_None.pkl",
        init_params=reset,
    ),
}

default_puzzle_size: int = 15
# fmt: on

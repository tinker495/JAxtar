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
from puzzle import (
    TSP,
    DotKnot,
    LightsOut,
    LightsOutHard,
    Maze,
    PancakeSorting,
    Puzzle,
    RubiksCube,
    RubiksCubeHard,
    RubiksCubeRandom,
    SlidePuzzle,
    SlidePuzzleHard,
    SlidePuzzleRandom,
    Sokoban,
    SokobanHard,
    TopSpin,
    TowerOfHanoi,
)
from puzzle.world_model import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModel_reversed,
    RubiksCubeWorldModel_test,
    RubiksCubeWorldModelOptimized,
    RubiksCubeWorldModelOptimized_reversed,
    RubiksCubeWorldModelOptimized_test,
    SokobanWorldModel,
    SokobanWorldModelOptimized,
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

default_puzzle_sizes: dict[str, int] = {
    "n-puzzle": 4,
    "n-puzzle-conv": 4,
    "n-puzzle-random": 4,
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
        path="puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_test": lambda **kwargs: RubiksCubeWorldModel_test(
        path="puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_reversed": lambda **kwargs: RubiksCubeWorldModel_reversed(
        path="puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_optimized": lambda **kwargs: RubiksCubeWorldModelOptimized(
        path="puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_test": lambda **kwargs: RubiksCubeWorldModelOptimized_test(
        path="puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_reversed": lambda **kwargs: RubiksCubeWorldModelOptimized_reversed(
        path="puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "sokoban_world_model": lambda **kwargs: SokobanWorldModel(
        path="puzzle/world_model/model/params/sokoban.pkl"
    ),
    "sokoban_world_model_optimized": lambda **kwargs: SokobanWorldModelOptimized(
        path="puzzle/world_model/model/params/sokoban_optimized.pkl"
    ),
}

puzzle_dict_hard: dict[str, Puzzle] = {
    "n-puzzle": SlidePuzzleHard,
    "n-puzzle-conv": SlidePuzzleHard,
    "lightsout": LightsOutHard,
    "lightsout-conv": LightsOutHard,
    "rubikscube": RubiksCubeHard,
    "sokoban": SokobanHard,
}

puzzle_heuristic_dict: dict[str, Heuristic] = {
    "n-puzzle": SlidePuzzleHeuristic,
    "n-puzzle-conv": SlidePuzzleHeuristic,
    "n-puzzle-random": SlidePuzzleHeuristic,
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
    "n-puzzle": lambda n, puzzle, reset: SlidePuzzleNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle_{n}.pkl",
        init_params=reset,
    ),
    "n-puzzle-random": lambda n, puzzle, reset: SlidePuzzleNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle-random_{n}.pkl",
        init_params=reset,
    ),
    "n-puzzle-conv": lambda n, puzzle, reset: SlidePuzzleConvNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/n-puzzle-conv_{n}.pkl",
        init_params=reset,
    ),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/lightsout_{n}.pkl",
        init_params=reset,
    ),
    "lightsout-conv": lambda n, puzzle, reset: LightsOutConvNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/lightsout-conv_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/rubikscube_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube-random": lambda n, puzzle, reset: RubiksCubeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/rubikscube-random_{n}.pkl",
        init_params=reset,
    ),
    "sokoban": lambda n, puzzle, reset: SokobanNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/sokoban_{n}.pkl",
        init_params=reset,
    ),
    "pancake": lambda n, puzzle, reset: PancakeNeuralHeuristic(
        puzzle=puzzle,
        path=f"heuristic/neuralheuristic/model/params/pancake_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_test": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_reversed": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_test": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_reversed": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/sokoban_world_model_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle=puzzle,
        path="heuristic/neuralheuristic/model/params/sokoban_world_model_optimized_None.pkl",
        init_params=reset,
    ),
}

puzzle_q_dict: dict[str, QFunction] = {
    "n-puzzle": SlidePuzzleQ,
    "n-puzzle-conv": SlidePuzzleQ,
    "n-puzzle-random": SlidePuzzleQ,
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
    "worldmodel": EmptyQFunction,
    "rubikscube_world_model": EmptyQFunction,
    "rubikscube_world_model_test": EmptyQFunction,
    "rubikscube_world_model_optimized": EmptyQFunction,
    "rubikscube_world_model_optimized_test": EmptyQFunction,
    "sokoban_world_model": EmptyQFunction,
    "sokoban_world_model_optimized": EmptyQFunction,
}

# nn option need to be callable, for loading model
puzzle_q_dict_nn: dict[str, callable] = {
    "n-puzzle": lambda n, puzzle, reset: SlidePuzzleNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle_{n}.pkl",
        init_params=reset,
    ),
    "n-puzzle-random": lambda n, puzzle, reset: SlidePuzzleNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle-random_{n}.pkl",
        init_params=reset,
    ),
    "n-puzzle-conv": lambda n, puzzle, reset: SlidePuzzleConvNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle-conv_{n}.pkl",
        init_params=reset,
    ),
    "n-puzzle-conv-random": lambda n, puzzle, reset: SlidePuzzleConvNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/n-puzzle-conv-random_{n}.pkl",
        init_params=reset,
    ),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/lightsout_{n}.pkl",
        init_params=reset,
    ),
    "lightsout-conv": lambda n, puzzle, reset: LightsOutConvNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/lightsout-conv_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/rubikscube_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube-random": lambda n, puzzle, reset: RubiksCubeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/rubikscube-random_{n}.pkl",
        init_params=reset,
    ),
    "sokoban": lambda n, puzzle, reset: SokobanNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/sokoban_{n}.pkl",
        init_params=reset,
    ),
    "pancake": lambda n, puzzle, reset: PancakeNeuralQ(
        puzzle=puzzle,
        path=f"qfunction/neuralq/model/params/pancake_{n}.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_test": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_reversed": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_test": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "rubikscube_world_model_optimized_reversed": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/sokoban_world_model_None.pkl",
        init_params=reset,
    ),
    "sokoban_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralQ(
        puzzle=puzzle,
        path="qfunction/neuralq/model/params/sokoban_world_model_optimized_None.pkl",
        init_params=reset,
    ),
}

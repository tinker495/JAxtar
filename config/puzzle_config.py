from heuristic import (
    DotKnotHeuristic,
    EmptyHeuristic,
    Heuristic,
    LightsOutHeuristic,
    MazeHeuristic,
    RubiksCubeHeuristic,
    SlidePuzzleHeuristic,
    SokobanHeuristic,
    TSPHeuristic,
)
from heuristic.neuralheuristic import (
    LightsOutNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
    WorldModelNeuralHeuristic,
)
from puzzle import (
    TSP,
    DotKnot,
    LightsOut,
    LightsOutHard,
    Maze,
    Puzzle,
    RubiksCube,
    RubiksCubeHard,
    SlidePuzzle,
    SlidePuzzleHard,
    Sokoban,
    SokobanHard,
)
from puzzle.world_model import (
    RubiksCubeWorldModel,
    RubiksCubeWorldModel_test,
    RubiksCubeWorldModelOptimized,
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
    QFunction,
    RubiksCubeQ,
    SlidePuzzleQ,
    SokobanQ,
)
from qfunction.neuralq import (
    LightsOutNeuralQ,
    RubiksCubeNeuralQ,
    SlidePuzzleNeuralQ,
    WorldModelNeuralQ,
)

default_puzzle_sizes: dict[str, int] = {
    "n-puzzle": 4,
    "lightsout": 7,
    "rubikscube": 3,
    "maze": 20,
    "tsp": 16,
    "dotknot": 8,
    "sokoban": 10,
    "rubikscube_world_model": None,
    "rubikscube_world_model_test": None,
    "rubikscube_world_model_optimized": None,
    "rubikscube_world_model_optimized_test": None,
    "sokoban_world_model": None,
    "sokoban_world_model_optimized": None,
}

puzzle_dict: dict[str, Puzzle] = {
    "n-puzzle": SlidePuzzle,
    "lightsout": LightsOut,
    "rubikscube": RubiksCube,
    "maze": Maze,
    "dotknot": DotKnot,
    "tsp": TSP,
    "sokoban": Sokoban,
    "rubikscube_world_model": lambda **kwargs: RubiksCubeWorldModel.load_model(
        "puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_test": lambda **kwargs: RubiksCubeWorldModel_test.load_model(
        "puzzle/world_model/model/params/rubikscube.pkl"
    ),
    "rubikscube_world_model_optimized": lambda **kwargs: RubiksCubeWorldModelOptimized.load_model(
        "puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_test": lambda **kwargs: RubiksCubeWorldModelOptimized_test.load_model(
        "puzzle/world_model/model/params/rubikscube_optimized.pkl"
    ),
    "sokoban_world_model": lambda **kwargs: SokobanWorldModel.load_model(
        "puzzle/world_model/model/params/sokoban.pkl"
    ),
    "sokoban_world_model_optimized": lambda **kwargs: SokobanWorldModelOptimized.load_model(
        "puzzle/world_model/model/params/sokoban_optimized.pkl"
    ),
}

puzzle_dict_hard: dict[str, Puzzle] = {
    "n-puzzle": SlidePuzzleHard,
    "lightsout": LightsOutHard,
    "rubikscube": RubiksCubeHard,
    "sokoban": SokobanHard,
}

puzzle_heuristic_dict: dict[str, Heuristic] = {
    "n-puzzle": SlidePuzzleHeuristic,
    "lightsout": LightsOutHeuristic,
    "rubikscube": RubiksCubeHeuristic,
    "maze": MazeHeuristic,
    "dotknot": DotKnotHeuristic,
    "tsp": TSPHeuristic,
    "sokoban": SokobanHeuristic,
    "rubikscube_world_model": EmptyHeuristic,
    "rubikscube_world_model_test": EmptyHeuristic,
    "rubikscube_world_model_optimized": EmptyHeuristic,
    "rubikscube_world_model_optimized_test": EmptyHeuristic,
    "sokoban_world_model": EmptyHeuristic,
    "sokoban_world_model_optimized": EmptyHeuristic,
}

# nn option need to be callable, for loading model
puzzle_heuristic_dict_nn: dict[str, callable] = {
    "n-puzzle": lambda n, puzzle, reset: SlidePuzzleNeuralHeuristic(puzzle)
    if reset
    else SlidePuzzleNeuralHeuristic.load_model(
        puzzle, f"heuristic/neuralheuristic/model/params/n-puzzle_{n}.pkl"
    ),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralHeuristic(puzzle)
    if reset
    else LightsOutNeuralHeuristic.load_model(
        puzzle, f"heuristic/neuralheuristic/model/params/lightsout_{n}.pkl"
    ),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralHeuristic(puzzle)
    if reset
    else RubiksCubeNeuralHeuristic.load_model(
        puzzle, f"heuristic/neuralheuristic/model/params/rubikscube_{n}.pkl"
    ),
    "rubikscube_world_model": lambda n, puzzle, reset: WorldModelNeuralHeuristic(puzzle)
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/rubikscube_world_model.pkl"
    ),
    "rubikscube_world_model_test": lambda n, puzzle, reset: WorldModelNeuralHeuristic(puzzle)
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/rubikscube_world_model.pkl"
    ),
    "rubikscube_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralHeuristic(puzzle)
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/rubikscube_world_model_optimized.pkl"
    ),
    "rubikscube_world_model_optimized_test": lambda n, puzzle, reset: WorldModelNeuralHeuristic(
        puzzle
    )
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/rubikscube_world_model_optimized.pkl"
    ),
    "sokoban_world_model": lambda n, puzzle, reset: WorldModelNeuralHeuristic(puzzle)
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/sokoban_world_model.pkl"
    ),
    "sokoban_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralHeuristic(puzzle)
    if reset
    else WorldModelNeuralHeuristic.load_model(
        puzzle, "puzzle/neuralheuristic/model/params/sokoban_world_model_optimized.pkl"
    ),
}

puzzle_q_dict: dict[str, QFunction] = {
    "n-puzzle": SlidePuzzleQ,
    "lightsout": LightsOutQ,
    "rubikscube": RubiksCubeQ,
    "maze": MazeQ,
    "dotknot": DotKnotQ,
    "tsp": TSPQ,
    "sokoban": SokobanQ,
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
    "n-puzzle": lambda n, puzzle, reset: SlidePuzzleNeuralQ(puzzle)
    if reset
    else SlidePuzzleNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/n-puzzle_{n}.pkl"),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralQ(puzzle)
    if reset
    else LightsOutNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/lightsout_{n}.pkl"),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralQ(puzzle)
    if reset
    else RubiksCubeNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/rubikscube_{n}.pkl"),
    "rubikscube_world_model": lambda n, puzzle, reset: WorldModelNeuralQ(puzzle)
    if reset
    else WorldModelNeuralQ.load_model(
        puzzle, "qfunction/neuralq/model/params/rubikscube_world_model.pkl"
    ),
    "rubikscube_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralQ(puzzle)
    if reset
    else WorldModelNeuralQ.load_model(
        puzzle, "qfunction/neuralq/model/params/rubikscube_world_model_optimized.pkl"
    ),
    "sokoban_world_model": lambda n, puzzle, reset: WorldModelNeuralQ(puzzle)
    if reset
    else WorldModelNeuralQ.load_model(
        puzzle, "qfunction/neuralq/model/params/sokoban_world_model.pkl"
    ),
    "sokoban_world_model_optimized": lambda n, puzzle, reset: WorldModelNeuralQ(puzzle)
    if reset
    else WorldModelNeuralQ.load_model(
        puzzle, "qfunction/neuralq/model/params/sokoban_world_model_optimized.pkl"
    ),
}

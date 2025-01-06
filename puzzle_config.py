from heuristic import (
    DotKnotHeuristic,
    EmptyHeuristic,
    LightsOutHeuristic,
    MazeHeuristic,
    RubiksCubeHeuristic,
    SlidePuzzleHeuristic,
)
from heuristic.neuralheuristic import (
    LightsOutNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
)
from puzzle import (
    TSP,
    DotKnot,
    LightsOut,
    LightsOutHard,
    Maze,
    RubiksCube,
    RubiksCubeHard,
    SlidePuzzle,
    SlidePuzzleHard,
)
from qfunction import (
    DotKnotQ,
    EmptyQFunction,
    LightsOutQ,
    MazeQ,
    RubiksCubeQ,
    SlidePuzzleQ,
)
from qfunction.neuralq import LightsOutNeuralQ, RubiksCubeNeuralQ, SlidePuzzleNeuralQ

default_puzzle_sizes = {
    "n-puzzle": 4,
    "lightsout": 7,
    "rubikscube": 3,
    "maze": 20,
    "dotknot": 7,
    "tsp": 10,
}

puzzle_dict = {
    "n-puzzle": lambda n: SlidePuzzle(n),
    "lightsout": lambda n: LightsOut(n),
    "rubikscube": lambda n: RubiksCube(n),
    "maze": lambda n: Maze(n),
    "dotknot": lambda n: DotKnot(n),
    "tsp": lambda n: TSP(n),
}

puzzle_dict_hard = {
    "n-puzzle": lambda n: SlidePuzzleHard(n),
    "lightsout": lambda n: LightsOutHard(n),
    "rubikscube": lambda n: RubiksCubeHard(n),
    "maze": lambda n: Maze(n),
    "dotknot": lambda n: DotKnot(n),
    "tsp": lambda n: TSP(n),
}

puzzle_heuristic_dict = {
    "n-puzzle": lambda puzzle: SlidePuzzleHeuristic(puzzle),
    "lightsout": lambda puzzle: LightsOutHeuristic(puzzle),
    "rubikscube": lambda puzzle: RubiksCubeHeuristic(puzzle),
    "maze": lambda puzzle: MazeHeuristic(puzzle),
    "dotknot": lambda puzzle: DotKnotHeuristic(puzzle),
    "tsp": lambda puzzle: EmptyHeuristic(puzzle),
}

puzzle_heuristic_dict_nn = {
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
}

puzzle_q_dict = {
    "n-puzzle": lambda puzzle: SlidePuzzleQ(puzzle),
    "lightsout": lambda puzzle: LightsOutQ(puzzle),
    "rubikscube": lambda puzzle: RubiksCubeQ(puzzle),
    "maze": lambda puzzle: MazeQ(puzzle),
    "dotknot": lambda puzzle: DotKnotQ(puzzle),
    "tsp": lambda puzzle: EmptyQFunction(puzzle),
}

puzzle_q_dict_nn = {
    "n-puzzle": lambda n, puzzle, reset: SlidePuzzleNeuralQ(puzzle)
    if reset
    else SlidePuzzleNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/n-puzzle_{n}.pkl"),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralQ(puzzle)
    if reset
    else LightsOutNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/lightsout_{n}.pkl"),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralQ(puzzle)
    if reset
    else RubiksCubeNeuralQ.load_model(puzzle, f"qfunction/neuralq/model/params/rubikscube_{n}.pkl"),
}

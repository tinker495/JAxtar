from puzzle.slidepuzzle import SlidePuzzle, SlidePuzzleHard
from puzzle.lightsout import LightsOut, LightsOutHard
from puzzle.rubikscube import RubiksCube, RubiksCubeHard
from puzzle.maze import Maze
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from heuristic.lightsout_heuristic import LightsOutHeuristic
from heuristic.rubikscube_heuristic import RubiksCubeHeuristic
from heuristic.maze_heuristic import MazeHeuristic
from heuristic.neuralheuristic.model.slidepuzzle_neuralheuristic import SlidePuzzleNeuralHeuristic
from heuristic.neuralheuristic.model.lightsout_neuralheuristic import LightsOutNeuralHeuristic
from heuristic.neuralheuristic.model.rubikscube_neuralheuristic import RubiksCubeNeuralHeuristic

default_puzzle_sizes = {
    "n-puzzle": 4,
    "lightsout": 7,
    "rubikscube": 3,
    "maze": 20
}

puzzle_dict = {
    "n-puzzle": lambda n: SlidePuzzle(n),
    "lightsout": lambda n: LightsOut(n),
    "rubikscube": lambda n: RubiksCube(n),
    "maze": lambda n: Maze(n)
}

puzzle_dict_hard = {
    "n-puzzle": lambda n: SlidePuzzleHard(n),
    "lightsout": lambda n: LightsOutHard(n),
    "rubikscube": lambda n: RubiksCubeHard(n),
    "maze": lambda n: Maze(n)
}

puzzle_heuristic_dict = {
    "n-puzzle": lambda puzzle: SlidePuzzleHeuristic(puzzle),
    "lightsout": lambda puzzle: LightsOutHeuristic(puzzle),
    "rubikscube": lambda puzzle: RubiksCubeHeuristic(puzzle),
    "maze": lambda puzzle: MazeHeuristic(puzzle)
}

puzzle_heuristic_dict_nn = {
    "n-puzzle": lambda n, puzzle, reset:  SlidePuzzleNeuralHeuristic(puzzle)
                                    if reset else SlidePuzzleNeuralHeuristic.load_model(puzzle, f"heuristic/neuralheuristic/model/params/n-puzzle_{n}.pkl"),
    "lightsout": lambda n, puzzle, reset: LightsOutNeuralHeuristic(puzzle) 
                                    if reset else LightsOutNeuralHeuristic.load_model(puzzle, f"heuristic/neuralheuristic/model/params/lightsout_{n}.pkl"),
    "rubikscube": lambda n, puzzle, reset: RubiksCubeNeuralHeuristic(puzzle)
                                    if reset else RubiksCubeNeuralHeuristic.load_model(puzzle, f"heuristic/neuralheuristic/model/params/rubikscube_{n}.pkl"),
    #"maze": lambda n: (Maze(n), MazeNeuralHeuristic(Maze(n)).distance)
}
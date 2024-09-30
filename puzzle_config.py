from puzzle.slidepuzzle import SlidePuzzle
from puzzle.lightsout import LightsOut
from puzzle.rubikscube import RubiksCube
from puzzle.maze import Maze
from heuristic.slidepuzzle_heuristic import SlidePuzzleHeuristic
from heuristic.DAVI.neuralheuristic.slidepuzzle_neuralheuristic import SlidePuzzleNeuralHeuristic
from heuristic.DAVI.neuralheuristic.lightsout_neuralheuristic import LightsOutNeuralHeuristic
from heuristic.DAVI.neuralheuristic.rubikscube_neuralheuristic import RubiksCubeNeuralHeuristic
from heuristic.lightsout_heuristic import LightsOutHeuristic
from heuristic.rubikscube_heuristic import RubiksCubeHeuristic
from heuristic.maze_heuristic import MazeHeuristic

default_puzzle_sizes = {
    "n-puzzle": 4,
    "lightsout": 7,
    "rubikscube": 3,
    "maze": 20
}

puzzle_dict = {
    "n-puzzle": lambda n: (SlidePuzzle(n), SlidePuzzleHeuristic(SlidePuzzle(n))),
    "lightsout": lambda n: (LightsOut(n), LightsOutHeuristic(LightsOut(n))),
    "rubikscube": lambda n: (RubiksCube(n), RubiksCubeHeuristic(RubiksCube(n))),
    "maze": lambda n: (Maze(n), MazeHeuristic(Maze(n)))
}

puzzle_dict_nn = {
    "n-puzzle": lambda n, reset: (SlidePuzzle(n), SlidePuzzleNeuralHeuristic(SlidePuzzle(n)) 
                                    if reset else SlidePuzzleNeuralHeuristic.load_model(SlidePuzzle(n), f"heuristic/DAVI/neuralheuristic/params/n-puzzle_{n}.pkl")),
    "lightsout": lambda n, reset: (LightsOut(n), LightsOutNeuralHeuristic(LightsOut(n)) 
                                    if reset else LightsOutNeuralHeuristic.load_model(LightsOut(n), f"heuristic/DAVI/neuralheuristic/params/lightsout_{n}.pkl")),
    "rubikscube": lambda n, reset: (RubiksCube(n), RubiksCubeNeuralHeuristic(RubiksCube(n))
                                    if reset else RubiksCubeNeuralHeuristic.load_model(RubiksCube(n), f"heuristic/DAVI/neuralheuristic/params/rubikscube_{n}.pkl")),
    #"maze": lambda n: (Maze(n), MazeNeuralHeuristic(Maze(n)).distance)
}
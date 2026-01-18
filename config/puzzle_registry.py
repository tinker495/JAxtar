from typing import Dict

from puxle import (
    PDDL,
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
    PDDLHeuristic,
    RubiksCubeHeuristic,
    SlidePuzzleHeuristic,
    SokobanHeuristic,
    TSPHeuristic,
)
from heuristic.neuralheuristic import (
    LightsOutConvNeuralHeuristic,
    LightsOutNeuralHeuristic,
    PancakeNeuralHeuristic,
    RubiksCubeHLGNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    RubiksCubeRandomHLGNeuralHeuristic,
    RubiksCubeRandomNeuralHeuristic,
    SlidePuzzleConvNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
    SokobanNeuralHeuristic,
    WorldModelNeuralHeuristic,
)
from qfunction import (
    PDDLQ,
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
    RubiksCubeHLGNeuralQ,
    RubiksCubeNeuralQ,
    RubiksCubeRandomHLGNeuralQ,
    RubiksCubeRandomNeuralQ,
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
        k_max=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_{size}.pkl",
            ),
        },
        eval_benchmark="slide15-deepcubea",
    ),
    "n-puzzle-random": PuzzleBundle(
        puzzle=SlidePuzzleRandom,
        k_max=500,
        heuristic=SlidePuzzleHeuristic,
        q_function=SlidePuzzleQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-random_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-random_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_{size}.pkl",
            ),
        },
    ),
    "lightsout": PuzzleBundle(
        puzzle=LightsOut,
        puzzle_hard=PuzzleConfig(callable=LightsOut, initial_shuffle=50),
        heuristic=LightsOutHeuristic,
        q_function=LightsOutQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout-conv_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout_{size}.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout-conv_{size}.pkl",
            ),
        },
        eval_options_configs={
            "default": EvalOptions(
                batch_size=1000,
                cost_weight=0.2,
                pop_ratio=float("inf"),
            )
        },
        search_options_configs={
            "default": SearchOptions(
                cost_weight=0.2,
                batch_size=1000,
                pop_ratio=float("inf"),
            )
        },
        eval_benchmark="lightsout-deepcubea",
    ),
    "rubikscube": PuzzleBundle(
        puzzle=PuzzleConfig(callable=RubiksCube),
        puzzle_hard=PuzzleConfig(callable=RubiksCube, initial_shuffle=100),
        k_max=26,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeHLGNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-hlg_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeHLGNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-hlg_{size}.pkl",
            ),
        },
        eval_benchmark="rubikscube-deepcubea",
    ),
    "rubikscube-random": PuzzleBundle(
        puzzle=PuzzleConfig(callable=RubiksCubeRandom),
        k_max=26,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeRandomNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-random_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeRandomHLGNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-random-hlg_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeRandomNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-random_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeRandomHLGNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-random-hlg_{size}.pkl",
            ),
        },
    ),
    "rubikscube-uqtm": PuzzleBundle(
        puzzle=PuzzleConfig(callable=RubiksCube, kwargs={"metric": "UQTM"}),
        puzzle_hard=PuzzleConfig(
            callable=RubiksCube, kwargs={"metric": "UQTM"}, initial_shuffle=100
        ),
        k_max=26,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeHLGNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm-hlg_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeHLGNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm-hlg_{size}.pkl",
            ),
        },
        eval_benchmark="rubikscube-santa333",
    ),
    "rubikscube-uqtm-random": PuzzleBundle(
        puzzle=PuzzleConfig(callable=RubiksCubeRandom, kwargs={"metric": "UQTM"}),
        puzzle_hard=PuzzleConfig(
            callable=RubiksCubeRandom, kwargs={"metric": "UQTM"}, initial_shuffle=100
        ),
        k_max=26,
        heuristic=RubiksCubeHeuristic,
        q_function=RubiksCubeQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeRandomNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm-random_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeRandomHLGNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm-random-hlg_{size}.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeRandomNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm-random_{size}.pkl",
            ),
            "hlg": NeuralCallableConfig(
                callable=RubiksCubeRandomHLGNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm-random-hlg_{size}.pkl",
            ),
        },
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
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SokobanNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/sokoban_{size}.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SokobanNeuralQ,
                param_path="qfunction/neuralq/model/params/sokoban_{size}.pkl",
            )
        },
        k_max=500,
    ),
    "pancake": PuzzleBundle(
        puzzle=PancakeSorting,
        heuristic=PancakeHeuristic,
        q_function=PancakeQ,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=PancakeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/pancake_{size}.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=PancakeNeuralQ,
                param_path="qfunction/neuralq/model/params/pancake_{size}.pkl",
            )
        },
    ),
    "hanoi": PuzzleBundle(puzzle=TowerOfHanoi),
    "topspin": PuzzleBundle(puzzle=TopSpin),
    "pddl_blocksworld": PuzzleBundle(
        puzzle=lambda: PDDL.from_preset("blocksworld", "bw-H-01"),
        heuristic=PDDLHeuristic,
        q_function=PDDLQ,
    ),
    "pddl_gripper": PuzzleBundle(
        puzzle=lambda: PDDL.from_preset("gripper", "gr-H-01"),
        heuristic=PDDLHeuristic,
        q_function=PDDLQ,
    ),
    "pddl_logistics": PuzzleBundle(
        puzzle=lambda: PDDL.from_preset("logistics", "lg-H-01"),
        heuristic=PDDLHeuristic,
        q_function=PDDLQ,
    ),
    "pddl_rovers": PuzzleBundle(
        puzzle=lambda: PDDL.from_preset("rovers", "rv-H-01"),
        heuristic=PDDLHeuristic,
        q_function=PDDLQ,
    ),
    "pddl_satellite": PuzzleBundle(
        puzzle=lambda: PDDL.from_preset("satellite", "st-H-01"),
        heuristic=PDDLHeuristic,
        q_function=PDDLQ,
    ),
    "rubikscube_world_model": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel,
            path="world_model_puzzle/model/params/rubikscube.pkl",
        ),
        k_max=30,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            )
        },
    ),
    "rubikscube_world_model_test": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel_test,
            path="world_model_puzzle/model/params/rubikscube.pkl",
        ),
        k_max=30,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            )
        },
    ),
    "rubikscube_world_model_reversed": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModel_reversed,
            path="world_model_puzzle/model/params/rubikscube.pkl",
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_None.pkl",
            )
        },
    ),
    "rubikscube_world_model_optimized": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
    ),
    "rubikscube_world_model_optimized_test": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized_test,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
    ),
    "rubikscube_world_model_optimized_reversed": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=RubiksCubeWorldModelOptimized_reversed,
            path="world_model_puzzle/model/params/rubikscube_optimized.pkl",
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_world_model_optimized_None.pkl",
            )
        },
    ),
    "sokoban_world_model": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=SokobanWorldModel, path="world_model_puzzle/model/params/sokoban.pkl"
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/sokoban_world_model_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/sokoban_world_model_None.pkl",
            )
        },
    ),
    "sokoban_world_model_optimized": PuzzleBundle(
        puzzle=WorldModelPuzzleConfig(
            callable=SokobanWorldModelOptimized,
            path="world_model_puzzle/model/params/sokoban_optimized.pkl",
        ),
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/sokoban_world_model_optimized_None.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=WorldModelNeuralQ,
                param_path="qfunction/neuralq/model/params/sokoban_world_model_optimized_None.pkl",
            )
        },
    ),
}

# --- Sized variants registered below for clarity and explicit selection ---


def _sized_bundle(
    base: PuzzleBundle,
    *,
    size: int,
    puzzle_cls,
    kwargs=dict(),
    hard_cls=None,
    hard_initial_shuffle=None,
) -> PuzzleBundle:
    """Create a sized variant of a base PuzzleBundle by binding size into PuzzleConfig.kwargs.

    This preserves other bundle fields (heuristic configs, q-function configs, etc.).
    """
    sized_puzzle = PuzzleConfig(callable=puzzle_cls, kwargs={"size": size, **kwargs})
    if hard_cls is not None:
        sized_puzzle_hard = PuzzleConfig(
            callable=hard_cls,
            kwargs={"size": size, **kwargs},
            initial_shuffle=hard_initial_shuffle,
        )
    else:
        sized_puzzle_hard = None

    return base.model_copy(
        update={
            "puzzle": sized_puzzle,
            "puzzle_hard": sized_puzzle_hard,
        }
    )


# n-puzzle families (3x3, 4x4, 5x5, 6x6, 7x7) with size-specific k_max defaults
_NP_KMAX = {3: 100, 4: 500, 5: 500, 6: 500, 7: 500}
_NP_BENCHMARKS = {
    4: "slide15-deepcubea",
    5: "slide24-deepcubea",
    6: "slide35-deepcubea",
    7: "slide48-deepcubea",
}
for _s in [3, 4, 5, 6, 7]:
    bundle = _sized_bundle(
        puzzle_bundles["n-puzzle"], size=_s, puzzle_cls=SlidePuzzle, hard_cls=None
    )
    bundle.k_max = _NP_KMAX[_s]
    bundle.eval_benchmark = _NP_BENCHMARKS.get(_s)
    puzzle_bundles[f"n-puzzle-{_s}"] = bundle

    bundle_r = _sized_bundle(
        puzzle_bundles["n-puzzle-random"], size=_s, puzzle_cls=SlidePuzzleRandom, hard_cls=None
    )
    bundle_r.k_max = _NP_KMAX[_s]
    bundle_r.eval_benchmark = _NP_BENCHMARKS.get(_s)
    puzzle_bundles[f"n-puzzle-random-{_s}"] = bundle_r

# LightsOut common sizes (5, 7) with same hard initial shuffle policy and size-specific k_max
_LO_KMAX = {5: 50, 7: 70}
for _s in [5, 7]:
    bundle_l = _sized_bundle(
        puzzle_bundles["lightsout"],
        size=_s,
        puzzle_cls=LightsOut,
        hard_cls=LightsOut,
        hard_initial_shuffle=50,
    )
    bundle_l.k_max = _LO_KMAX[_s]
    if _s == 7:
        bundle_l.eval_benchmark = "lightsout-deepcubea"
    puzzle_bundles[f"lightsout-{_s}"] = bundle_l


# Rubik's Cube size 3 with size-specific k_max
_RC_KMAX = {3: 26, 4: 45, 5: 65}
_RC_UQTM_BENCHMARKS = {
    3: "rubikscube-santa333",
    4: "rubikscube-santa444",
    5: "rubikscube-santa555",
}
for _s in [3, 4, 5]:
    bundle_rc = _sized_bundle(
        puzzle_bundles["rubikscube"],
        size=_s,
        puzzle_cls=RubiksCube,
        hard_cls=RubiksCube,
        hard_initial_shuffle=120,
    )
    bundle_rc.k_max = _RC_KMAX[_s]
    if _s == 3:
        bundle_rc.eval_benchmark = "rubikscube-deepcubea"
    puzzle_bundles[f"rubikscube-{_s}"] = bundle_rc

    bundle_rcr = _sized_bundle(
        puzzle_bundles["rubikscube-random"], size=_s, puzzle_cls=RubiksCubeRandom, hard_cls=None
    )
    bundle_rcr.k_max = _RC_KMAX[_s]
    puzzle_bundles[f"rubikscube-random-{_s}"] = bundle_rcr

    bundle_rcu = _sized_bundle(
        puzzle_bundles["rubikscube-uqtm"],
        size=_s,
        puzzle_cls=RubiksCube,
        kwargs={"metric": "UQTM"},
        hard_cls=RubiksCube,
        hard_initial_shuffle=120,
    )
    bundle_rcu.k_max = _RC_KMAX[_s]
    bundle_rcu.eval_benchmark = _RC_UQTM_BENCHMARKS.get(_s)
    puzzle_bundles[f"rubikscube-uqtm-{_s}"] = bundle_rcu

    bundle_rcur = _sized_bundle(
        puzzle_bundles["rubikscube-uqtm-random"],
        size=_s,
        puzzle_cls=RubiksCubeRandom,
        kwargs={"metric": "UQTM"},
        hard_cls=None,
    )
    bundle_rcur.k_max = _RC_KMAX[_s]
    puzzle_bundles[f"rubikscube-uqtm-random-{_s}"] = bundle_rcur

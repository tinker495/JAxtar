from puxle.benchmark import (
    LightsOutDeepCubeABenchmark,
    RubiksCubeDeepCubeABenchmark,
    RubiksCubeDeepCubeAHardBenchmark,
    RubiksCubeSanta333Benchmark,
    RubiksCubeSanta444Benchmark,
    RubiksCubeSanta555Benchmark,
    RubiksCubeSanta666Benchmark,
    SlidePuzzleDeepCubeA15HardBenchmark,
    SlidePuzzleDeepCubeA24Benchmark,
    SlidePuzzleDeepCubeA35Benchmark,
    SlidePuzzleDeepCubeA48Benchmark,
    SlidePuzzleDeepCubeABenchmark,
)

from heuristic.neuralheuristic import (
    LightsOutConvNeuralHeuristic,
    LightsOutNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    SlidePuzzleConvNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
)
from qfunction.neuralq import (
    LightsOutConvNeuralQ,
    LightsOutNeuralQ,
    RubiksCubeNeuralQ,
    SlidePuzzleConvNeuralQ,
    SlidePuzzleNeuralQ,
)

from .pydantic_models import BenchmarkBundle, NeuralCallableConfig

benchmark_bundles: dict[str, BenchmarkBundle] = {
    "rubikscube-deepcubea": BenchmarkBundle(
        benchmark=RubiksCubeDeepCubeABenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_4M.pkl",
            ),
            "davi_lt01": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_davi_lt01.pkl",
            ),
            "asym_logcosh": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_asym_logcosh.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_4M.pkl",
            ),
            "qlearning_lt01": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_qlearning_lt01.pkl",
            ),
        },
    ),
    "rubikscube-hard-deepcubea": BenchmarkBundle(
        benchmark=RubiksCubeDeepCubeAHardBenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_4M.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_4M.pkl",
            ),
        },
    ),
    "rubikscube-santa333": BenchmarkBundle(
        benchmark=RubiksCubeSanta333Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_3_4M.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_3.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_3_4M.pkl",
            ),
        },
    ),
    "rubikscube-santa444": BenchmarkBundle(
        benchmark=RubiksCubeSanta444Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_4.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_4_4M.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_4.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_4_4M.pkl",
            ),
        },
    ),
    "rubikscube-santa555": BenchmarkBundle(
        benchmark=RubiksCubeSanta555Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_5.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_5_4M.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_5.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_5_4M.pkl",
            ),
        },
    ),
    "rubikscube-santa666": BenchmarkBundle(
        benchmark=RubiksCubeSanta666Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_6.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_6_4M.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_6.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_6_4M.pkl",
            ),
        },
    ),
    "slide15-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeABenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_4.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_4.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_4.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_4.pkl",
            ),
        },
    ),
    "slide15-hard-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA15HardBenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_4.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_4.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_4.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_4.pkl",
            ),
        },
    ),
    "slide24-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA24Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_5.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_5.pkl",
            )
        },
    ),
    "slide35-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA35Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_6.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_6.pkl",
            )
        },
    ),
    "slide48-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA48Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_7.pkl",
            )
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_7.pkl",
            )
        },
    ),
    "lightsout-deepcubea": BenchmarkBundle(
        benchmark=LightsOutDeepCubeABenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout_7.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout-conv_7.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout_7.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout-conv_7.pkl",
            ),
        },
    ),
}

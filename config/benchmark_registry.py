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
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_4M_v2.pkl",
            ),
        },
    ),
    "rubikscube-hard-deepcubea": BenchmarkBundle(
        benchmark=RubiksCubeDeepCubeAHardBenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube_3_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube_3_4M_v2.pkl",
            ),
        },
    ),
    "rubikscube-santa333": BenchmarkBundle(
        benchmark=RubiksCubeSanta333Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_3_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_3_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_3_4M_v2.pkl",
            ),
        },
    ),
    "rubikscube-santa444": BenchmarkBundle(
        benchmark=RubiksCubeSanta444Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_4_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_4_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_4_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_4_4M_v2.pkl",
            ),
        },
    ),
    "rubikscube-santa555": BenchmarkBundle(
        benchmark=RubiksCubeSanta555Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_5_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_5_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_5_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_5_4M_v2.pkl",
            ),
        },
    ),
    "rubikscube-santa666": BenchmarkBundle(
        benchmark=RubiksCubeSanta666Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_6_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/rubikscube-uqtm_6_4M_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_6_v2.pkl",
            ),
            "4M": NeuralCallableConfig(
                callable=RubiksCubeNeuralQ,
                param_path="qfunction/neuralq/model/params/rubikscube-uqtm_6_4M_v2.pkl",
            ),
        },
    ),
    "slide15-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeABenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_4_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_4_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_4_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_4_v2.pkl",
            ),
        },
    ),
    "slide15-hard-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA15HardBenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_4_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle-conv_4_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_4_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=SlidePuzzleConvNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle-conv_4_v2.pkl",
            ),
        },
    ),
    "slide24-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA24Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_5_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_5_v2.pkl",
            ),
        },
    ),
    "slide35-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA35Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_6_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_6_v2.pkl",
            ),
        },
    ),
    "slide48-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA48Benchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/n-puzzle_7_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=SlidePuzzleNeuralQ,
                param_path="qfunction/neuralq/model/params/n-puzzle_7_v2.pkl",
            ),
        },
    ),
    "lightsout-deepcubea": BenchmarkBundle(
        benchmark=LightsOutDeepCubeABenchmark,
        heuristic_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout_7_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralHeuristic,
                param_path="heuristic/neuralheuristic/model/params/lightsout-conv_7_v2.pkl",
            ),
        },
        q_function_nn_configs={
            "default": NeuralCallableConfig(
                callable=LightsOutNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout_7_v2.pkl",
            ),
            "conv": NeuralCallableConfig(
                callable=LightsOutConvNeuralQ,
                param_path="qfunction/neuralq/model/params/lightsout-conv_7_v2.pkl",
            ),
        },
    ),
}

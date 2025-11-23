from puxle.benchmark import (
    LightsOutDeepCubeABenchmark,
    RubiksCubeDeepCubeABenchmark,
    SlidePuzzleDeepCubeA15HardBenchmark,
    SlidePuzzleDeepCubeA24Benchmark,
    SlidePuzzleDeepCubeA35Benchmark,
    SlidePuzzleDeepCubeA48Benchmark,
    SlidePuzzleDeepCubeABenchmark,
)

from heuristic.neuralheuristic import (
    LightsOutNeuralHeuristic,
    RubiksCubeNeuralHeuristic,
    SlidePuzzleNeuralHeuristic,
)
from qfunction.neuralq import LightsOutNeuralQ, RubiksCubeNeuralQ, SlidePuzzleNeuralQ

from .pydantic_models import BenchmarkBundle, EvalOptions, NeuralCallableConfig

benchmark_bundles: dict[str, BenchmarkBundle] = {
    "rubikscube-deepcubea": BenchmarkBundle(
        benchmark=RubiksCubeDeepCubeABenchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/rubikscube_3.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=RubiksCubeNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/rubikscube_3.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "slide15-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeABenchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/n-puzzle_4.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/n-puzzle_4.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "slide15-hard-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA15HardBenchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/n-puzzle_4.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/n-puzzle_4.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "slide24-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA24Benchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/n-puzzle_5.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/n-puzzle_5.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "slide35-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA35Benchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/n-puzzle_6.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/n-puzzle_6.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "slide48-deepcubea": BenchmarkBundle(
        benchmark=SlidePuzzleDeepCubeA48Benchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/n-puzzle_7.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=SlidePuzzleNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/n-puzzle_7.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=10000,
            cost_weight=0.6,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
    "lightsout-deepcubea": BenchmarkBundle(
        benchmark=LightsOutDeepCubeABenchmark,
        heuristic_nn_config=NeuralCallableConfig(
            callable=LightsOutNeuralHeuristic,
            param_paths={"default": "heuristic/neuralheuristic/model/params/lightsout_7.pkl"},
        ),
        q_function_nn_config=NeuralCallableConfig(
            callable=LightsOutNeuralQ,
            param_paths={"default": "qfunction/neuralq/model/params/lightsout_7.pkl"},
        ),
        eval_options=EvalOptions(
            batch_size=1000,
            cost_weight=0.2,
            pop_ratio=float("inf"),
            num_eval=-1,
            early_stop_patience=10,
            early_stop_threshold=0.1,
        ),
    ),
}

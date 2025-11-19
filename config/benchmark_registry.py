from puxle.benchmark import RubiksCubeDeepCubeABenchmark

from heuristic.neuralheuristic import RubiksCubeNeuralHeuristic
from qfunction.neuralq import RubiksCubeNeuralQ

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
}

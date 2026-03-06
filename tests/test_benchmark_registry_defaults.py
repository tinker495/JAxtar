from config.benchmark_registry import benchmark_bundles
from heuristic.neuralheuristic import SlidePuzzleConvNeuralHeuristic


def test_slide15_benchmark_default_heuristic_uses_available_conv_checkpoint():
    """Slide15 benchmarks should default to the published conv heuristic checkpoint."""
    config = benchmark_bundles["slide15-deepcubea"].heuristic_nn_configs["default"]

    assert config.callable is SlidePuzzleConvNeuralHeuristic
    assert config.param_path == "heuristic/neuralheuristic/model/params/n-puzzle-conv_4.pkl"


def test_slide15_hard_benchmark_default_heuristic_uses_available_conv_checkpoint():
    """Slide15-hard benchmarks should share the same available conv heuristic checkpoint."""
    config = benchmark_bundles["slide15-hard-deepcubea"].heuristic_nn_configs["default"]

    assert config.callable is SlidePuzzleConvNeuralHeuristic
    assert config.param_path == "heuristic/neuralheuristic/model/params/n-puzzle-conv_4.pkl"

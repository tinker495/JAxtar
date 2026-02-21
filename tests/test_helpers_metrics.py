from helpers.metrics import calculate_benchmark_metrics, calculate_heuristic_metrics


def test_calculate_heuristic_metrics_prefers_optimal_path_results_when_available():
    results = [
        {
            "path_analysis": {"actual": [3, 4], "estimated": [3, 5]},
            "used_optimal_path_for_analysis": True,
        },
        {
            "path_analysis": {"actual": [2], "estimated": [1]},
        },
    ]

    out = calculate_heuristic_metrics(results)

    assert out is not None
    assert out["has_optimal_path_used"] is True
    assert out["r_squared"] == -1.0
    assert out["ccc"] == 2 / 3


def test_calculate_heuristic_metrics_uses_all_results_without_optimal_path_flag():
    results = [
        {
            "path_analysis": {"actual": [1, 3], "estimated": [2, 2]},
        },
        {
            "path_analysis": {"actual": [1, 5], "estimated": [2, 6]},
        },
    ]

    out = calculate_heuristic_metrics(results)

    assert out is not None
    assert out["has_optimal_path_used"] is False
    assert out["r_squared"] < 1.0


def test_calculate_heuristic_metrics_returns_none_for_insufficient_data():
    results = [{"path_analysis": {"actual": [1], "estimated": [2]}}]

    assert calculate_heuristic_metrics(results) is None


def test_calculate_heuristic_metrics_handles_zero_variance_gracefully():
    results = [
        {
            "path_analysis": {"actual": [4, 4], "estimated": [4, 4]},
        }
    ]

    out = calculate_heuristic_metrics(results)

    assert out is not None
    assert out["r_squared"] == 1.0
    assert out["ccc"] == 1.0


def test_calculate_benchmark_metrics_aggregates_cost_and_action_gaps():
    results = [
        {
            "solved": True,
            "benchmark_has_optimal_action_sequence": True,
            "benchmark_optimal_path_cost": 12,
            "path_cost": 14,
            "benchmark_optimal_action_count": 5,
            "path_action_count": 7,
            "matches_optimal_path": True,
        },
        {
            "solved": True,
            "benchmark_has_optimal_action_sequence": True,
            "benchmark_optimal_path_cost": 10,
            "path_cost": 10,
            "benchmark_optimal_action_count": 6,
            "path_action_count": 6,
            "matches_optimal_path": False,
        },
        {"solved": False},
    ]

    out = calculate_benchmark_metrics(results)

    assert out["avg_optimal_cost"] == 11.0
    assert out["avg_path_cost"] == 12.0
    assert out["avg_cost_gap"] == 1.0
    assert out["solved_with_optimal_cost"] == 2
    assert out["avg_optimal_actions"] == 5.5
    assert out["avg_path_actions"] == 6.5
    assert out["avg_action_gap"] == 1.0
    assert out["solved_with_optimal_length"] == 2
    assert out["exact_optimal_path_rate"] == 0.5
    assert out["exact_optimal_path_count"] == 1


def test_calculate_benchmark_metrics_empty_when_no_relevant_fields():
    assert calculate_benchmark_metrics([{"solved": False}, {"seed": 1}]) == {}

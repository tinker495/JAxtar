from typing import Dict, List, Optional

import numpy as np


def calculate_heuristic_metrics(results: List[Dict]) -> Optional[Dict[str, float]]:
    all_actual_dists = []
    all_estimated_dists = []
    has_optimal_path_used = False

    # Include all results that have path analysis data, regardless of whether search solved it
    # (e.g. if we used optimal path from benchmark for analysis)
    results_with_analysis = [r for r in results if r.get("path_analysis")]

    # Prefer optimal-path analyses exclusively when present to avoid mixing reference and suboptimal search paths.
    optimal_path_results = [
        r for r in results_with_analysis if r.get("used_optimal_path_for_analysis")
    ]

    if optimal_path_results:
        target_results = optimal_path_results
        has_optimal_path_used = True
    else:
        target_results = results_with_analysis

    for r in target_results:
        if r.get("path_analysis"):
            analysis_data = r["path_analysis"]
            if analysis_data.get("actual") and analysis_data.get("estimated"):
                all_actual_dists.extend(analysis_data["actual"])
                all_estimated_dists.extend(analysis_data["estimated"])

    # Need at least 2 points to compute correlation
    if len(all_actual_dists) <= 1:
        return None

    y_true = np.array(all_actual_dists)
    y_pred = np.array(all_estimated_dists)

    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    if ss_total > 0:
        r_squared = 1 - (ss_residual / ss_total)
    else:
        # If total variance is zero, R-squared is 1 if error is also zero, else undefined (or 0).
        r_squared = 1.0 if ss_residual == 0 else 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    # Handle cases where std is zero, which would make corrcoef return NaN
    if std_true == 0 or std_pred == 0:
        pearson_corr = 1.0 if std_true == std_pred else 0.0
    else:
        corr_matrix = np.corrcoef(y_true, y_pred)
        pearson_corr = corr_matrix[0, 1]

    numerator = 2 * pearson_corr * std_true * std_pred
    denominator = std_true**2 + std_pred**2 + (mean_true - mean_pred) ** 2

    if denominator > 0:
        ccc = numerator / denominator
    else:
        # Denominator is zero only if stds are zero and means are equal,
        # which implies perfect concordance.
        ccc = 1.0

    return {"r_squared": r_squared, "ccc": ccc, "has_optimal_path_used": has_optimal_path_used}


def calculate_benchmark_metrics(results: List[Dict]) -> Dict[str, float]:
    metrics = {}

    solved_with_opt_cost = [
        r
        for r in results
        if r.get("solved")
        and r.get("benchmark_has_optimal_action_sequence")
        and r.get("benchmark_optimal_path_cost") is not None
        and r.get("path_cost") is not None
    ]

    if solved_with_opt_cost:
        avg_optimal = float(
            np.mean([r["benchmark_optimal_path_cost"] for r in solved_with_opt_cost])
        )
        path_costs = [r["path_cost"] for r in solved_with_opt_cost]
        avg_path_cost = float(np.mean(path_costs))
        cost_gap = avg_path_cost - avg_optimal
        metrics.update(
            {
                "avg_optimal_cost": avg_optimal,
                "avg_path_cost": avg_path_cost,
                "avg_cost_gap": cost_gap,
                "solved_with_optimal_cost": len(solved_with_opt_cost),
            }
        )

    solved_with_opt_length = [
        r
        for r in results
        if r.get("solved")
        and r.get("benchmark_optimal_action_count") is not None
        and r.get("path_action_count") is not None
    ]

    if solved_with_opt_length:
        avg_opt_actions = float(
            np.mean([r["benchmark_optimal_action_count"] for r in solved_with_opt_length])
        )
        avg_path_actions = float(np.mean([r["path_action_count"] for r in solved_with_opt_length]))
        action_gap = avg_path_actions - avg_opt_actions
        metrics.update(
            {
                "avg_optimal_actions": avg_opt_actions,
                "avg_path_actions": avg_path_actions,
                "avg_action_gap": action_gap,
                "solved_with_optimal_length": len(solved_with_opt_length),
            }
        )

    # Verification-based optimality applies even when no reference action sequence is provided.
    verification_matches = [
        r["matches_optimal_path"] for r in results if r.get("matches_optimal_path") is not None
    ]
    if verification_matches:
        verified_exact_matches = sum(1 for m in verification_matches if m)
        metrics["exact_optimal_path_rate"] = verified_exact_matches / len(verification_matches)
        metrics["exact_optimal_path_count"] = verified_exact_matches

    return metrics

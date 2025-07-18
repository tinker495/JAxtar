from typing import Dict, List, Optional

import numpy as np


def calculate_heuristic_metrics(results: List[Dict]) -> Optional[Dict[str, float]]:
    """Calculates R-squared and CCC for heuristic accuracy from evaluation results."""
    all_actual_dists = []
    all_estimated_dists = []
    solved_results = [r for r in results if r.get("solved")]
    for r in solved_results:
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

    # --- R-squared Calculation ---
    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Handle case where variance of true values is zero
    if ss_total > 0:
        r_squared = 1 - (ss_residual / ss_total)
    else:
        # If total variance is zero, R-squared is 1 if error is also zero, else undefined (or 0).
        r_squared = 1.0 if ss_residual == 0 else 0.0

    # --- Concordance Correlation Coefficient (CCC) Calculation ---
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

    # Handle case where denominator is zero
    if denominator > 0:
        ccc = numerator / denominator
    else:
        # Denominator is zero only if stds are zero and means are equal,
        # which implies perfect concordance.
        ccc = 1.0

    return {"r_squared": r_squared, "ccc": ccc}

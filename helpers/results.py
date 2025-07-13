import json
from pathlib import Path

import pandas as pd

from helpers.util import convert_to_serializable_dict


def save_evaluation_results(
    results: list[dict], run_dir: Path, config: dict, save_per_pop_ratio: bool = False
):
    """Saves evaluation results, config, and path states to files."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        serializable_config = convert_to_serializable_dict(config)
        json.dump(serializable_config, f, indent=4)

    # Save combined results
    pd.DataFrame(results).to_csv(run_dir / "results.csv", index=False)

    # Save path_states if present
    path_states_dir = run_dir / "path_states"
    path_states_dir.mkdir(exist_ok=True)
    for r in results:
        if r.get("path_analysis"):
            states = r["path_analysis"]["states"]
            if "pop_ratio" in r:
                filename = f"{r['seed']}_pr_{str(r['pop_ratio']).replace('.', '_')}.npz"
            else:
                filename = f"{r['seed']}.npz"
            states.save(path_states_dir / filename)

    if save_per_pop_ratio and "pop_ratio" in pd.DataFrame(results).columns:
        df = pd.DataFrame(results)
        for pr, group in df.groupby("pop_ratio"):
            group.to_csv(run_dir / f"results_pr_{str(pr).replace('.', '_')}.csv", index=False)

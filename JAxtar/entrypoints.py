"""Console entry points for JAxtar's checkout-local commands."""

import os
import sys
from functools import partial
from importlib import import_module
from pathlib import Path


def _run(module_name: str, command_name: str) -> None:
    project_root_path = Path(__file__).resolve().parent.parent
    if (
        not (project_root_path / "pyproject.toml").is_file()
        or not (project_root_path / "cli" / "main.py").is_file()
    ):
        raise SystemExit(
            "JAxtar CLI commands are checkout-only; " "run them from a JAxtar source checkout."
        )

    project_root = str(project_root_path)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    getattr(import_module(module_name), command_name)()


cli = partial(_run, "cli.main", "cli")
astar = partial(_run, "cli.commands", "astar")
astar_d = partial(_run, "cli.commands", "astar_d")
beam = partial(_run, "cli.commands", "beam")
benchmark = partial(_run, "cli.benchmark_commands", "benchmark")
bi_astar = partial(_run, "cli.commands", "bi_astar")
bi_astar_d = partial(_run, "cli.commands", "bi_astar_d")
bi_qstar = partial(_run, "cli.commands", "bi_qstar")
distance_train = partial(_run, "cli.train_commands", "distance_train")
id_astar = partial(_run, "cli.commands", "id_astar")
id_qstar = partial(_run, "cli.commands", "id_qstar")
qbeam = partial(_run, "cli.commands", "qbeam")
qstar = partial(_run, "cli.commands", "qstar")

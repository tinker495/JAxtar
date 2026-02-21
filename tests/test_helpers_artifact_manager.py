from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from helpers.artifact_manager import ArtifactManager


def test_save_config_writes_readable_serialized_json(tmp_path):
    run_dir = tmp_path / "run"

    manager = ArtifactManager(run_dir)
    manager.save_config({"callable": lambda x: x, "type_obj": int})

    payload = pd.read_json(run_dir / "config.json", typ="series")
    assert payload["callable"].startswith("<function")
    assert payload["type_obj"] == "int"


def test_save_results_writes_csv_for_results_records(tmp_path):
    manager = ArtifactManager(tmp_path / "run")
    manager.save_results([{"seed": 1, "solved": True}, {"seed": 2, "solved": False}])

    loaded = pd.read_csv(tmp_path / "run" / "results.csv")
    assert list(loaded.columns) == ["seed", "solved"]
    assert loaded.shape == (2, 2)


class _FakeState:
    def __init__(self, path_log: list[Path]):
        self.path_log = path_log

    def save(self, path: Path):
        self.path_log.append(path)
        path.write_text("state")


def test_save_path_states_serializes_state_payloads_per_seed(tmp_path):
    state_logs = []

    manager = ArtifactManager(tmp_path / "run")
    manager.save_path_states(
        [
            {"seed": 1, "path_analysis": {"states": _FakeState(state_logs)}},
            {"seed": 2, "path_analysis": {"states": _FakeState(state_logs)}},
            {"seed": 3, "no_path": True},
        ]
    )

    assert len(state_logs) == 2
    assert (tmp_path / "run" / "path_states" / "1.npz").exists()
    assert (tmp_path / "run" / "path_states" / "2.npz").exists()


class _FakeLogger:
    def __init__(self):
        self.figures = []
        self.scalars = []

    def log_figure(self, tag, fig, step):
        self.figures.append((tag, step))

    def log_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def test_save_and_log_plot_and_scalar_are_forwarded_to_logger(tmp_path):
    logger = _FakeLogger()
    run_dir = tmp_path / "run"
    manager = ArtifactManager(run_dir, logger=logger, step=3, log_namespace="case")

    fig = plt.figure()
    fig_num = fig.number
    manager.save_and_log_plot("tree", fig, sub_dir="plots")
    manager.log_scalar("accuracy", 0.89)

    assert (run_dir / "plots" / "tree.png").exists()
    assert logger.figures == [("Comparison/case/tree", 3)]
    assert logger.scalars == [("Comparison/case/accuracy", 0.89, 3)]
    assert not plt.fignum_exists(fig_num)

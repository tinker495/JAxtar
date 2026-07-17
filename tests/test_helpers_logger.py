from helpers.logger import AimLogger


class _AimRunStub:
    def __init__(self):
        self.params = {}

    def set(self, key, value):
        self.params[key] = value

    def track(self, value, *, name):
        if isinstance(value, dict) and name is not None:
            raise ValueError("'name' should be None when tracking values dictionary.")


def test_aim_logger_records_artifact_as_run_metadata(tmp_path):
    source = tmp_path / "heuristic_final.pkl"
    source.write_bytes(b"model")
    logger = object.__new__(AimLogger)
    logger.log_dir = str(tmp_path / "run")
    logger.aim_run = _AimRunStub()

    logger.log_artifact(str(source), "heuristic_final", "model")

    saved = tmp_path / "run" / "artifacts" / "model" / "heuristic_final"
    assert saved.read_bytes() == b"model"
    assert logger.aim_run.params == {
        ("artifacts", "model", "heuristic_final"): {
            "name": "heuristic_final",
            "type": "model",
            "path": str(saved),
            "original_path": str(source),
            "size": 5,
        }
    }

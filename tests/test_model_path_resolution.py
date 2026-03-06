from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from flax.traverse_util import flatten_dict
from huggingface_hub.errors import EntryNotFoundError
from puxle import SlidePuzzle

from heuristic.neuralheuristic.model.slidepuzzle_neuralheuristic import (
    SlidePuzzleConvNeuralHeuristic,
)
from neural_util.param_manager import load_params_with_metadata, merge_params
from neural_util.util import download_model, resolve_model_path


def test_resolve_model_path_uses_existing_non_v2_checkpoint():
    """Resolve local checkpoint aliases by falling back from _v2 to the existing file."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        requested = (
            root / "heuristic" / "neuralheuristic" / "model" / "params" / "n-puzzle_4_v2.pkl"
        )
        fallback = requested.with_name("n-puzzle_4.pkl")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_bytes(b"checkpoint")

        assert resolve_model_path(str(requested)) == str(fallback)


def test_download_model_falls_back_to_non_v2_artifact_name():
    """Download retries the non-_v2 alias when the requested artifact name is missing."""
    requested = "qfunction/neuralq/model/params/n-puzzle_4_v2.pkl"
    fallback = "qfunction/neuralq/model/params/n-puzzle_4.pkl"
    download_calls: list[str] = []

    def _fake_download(*, filename: str, **kwargs):
        download_calls.append(filename)
        if filename.endswith("_v2.pkl"):
            raise EntryNotFoundError(filename)
        return "/tmp/fallback"

    with patch("neural_util.util.huggingface_hub.hf_hub_download", side_effect=_fake_download):
        with patch("neural_util.util.resolve_model_path", side_effect=lambda filename: filename):
            assert download_model(requested) == fallback

    assert download_calls == [requested, fallback]


def test_merge_params_accepts_legacy_slidepuzzle_conv_resblock_layout():
    """Legacy slidepuzzle conv checkpoints should merge into the current ResBlock layout."""
    puzzle = SlidePuzzle(size=4)
    current_model = SlidePuzzleConvNeuralHeuristic(puzzle, init_params=True)
    old_params, _ = load_params_with_metadata(
        "heuristic/neuralheuristic/model/params/n-puzzle-conv_4.pkl"
    )

    merged = merge_params(current_model.params, old_params)

    merged_keys = set(flatten_dict(merged).keys())
    current_keys = set(flatten_dict(current_model.params).keys())

    assert current_keys == merged_keys

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import heuristic.neuralheuristic.target_dataset_builder as heuristic_builder
import qfunction.neuralq.target_dataset_builder as qfunction_builder
from cli.options import dist_train_options
from cli.train_commands.dist_train_command import heuristic_train_command


class _Puzzle:
    SolveConfig = int
    State = int


def test_training_labels_route_and_validate(monkeypatch):
    builders = (
        (heuristic_builder, heuristic_builder.get_heuristic_dataset_builder, "_get_datasets"),
        (
            qfunction_builder,
            qfunction_builder.get_qfunction_dataset_builder,
            "_get_datasets_with_policy",
        ),
    )
    for module, builder, base_name in builders:
        monkeypatch.setattr(
            module,
            "prepare_shuffled_path_sampling",
            lambda **kwargs: (1, 1, 1, object()),
        )
        monkeypatch.setattr(module, "xtructure_dataclass", lambda cls: cls)
        monkeypatch.setattr(module, "wrap_dataset_runner", lambda **kwargs: kwargs)

        expected_by_label = {
            "td": base_name,
            "diffusion": "_get_datasets_with_diffusion_distance",
            "diffusion_mixture": "_get_datasets_with_diffusion_distance_mixture",
        }
        for label, expected in expected_by_label.items():
            runner_config = builder(_Puzzle(), lambda: None, object(), 1, 1, 1, label=label)
            assert runner_config["diffusion_get_datasets"].func is getattr(module, expected)

        with pytest.raises(ValueError, match="Unknown training label"):
            builder(None, None, None, 1, 1, 1, label="unknown")


@pytest.mark.parametrize(("override", "expected"), [(None, "diffusion"), ("td", "td")])
def test_cli_label_overrides_preset_only_when_provided(override, expected):
    def command(**kwargs):
        return kwargs["train_options"]

    wrapped = dist_train_options(preset_category="heuristic_train")(command)
    options = wrapped(
        puzzle_bundle=SimpleNamespace(k_max=1),
        k_max=None,
        preset="diffusion_distance",
        label=override,
    )

    assert options.label == expected


def test_training_label_help_shows_effective_default():
    result = CliRunner().invoke(heuristic_train_command, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Default: 'td'." in " ".join(result.output.split())

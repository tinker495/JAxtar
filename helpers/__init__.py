from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "summaries",
    "plots",
    "config_printer",
    "print_config",
    "human_format",
    "heuristic_dist_format",
    "qfunction_dist_format",
    "human_format_to_float",
    "img_to_colored_str",
    "AimLogger",
    "BaseLogger",
    "NoOpLogger",
    "TensorboardLogger",
    "create_logger",
    "calculate_heuristic_metrics",
    "RichProgressBar",
    "tqdm",
    "trange",
    "convert_to_serializable_dict",
    "flatten_dict",
    "make_hashable",
    "display_value",
    "vmapping_search",
    "vmapping_init_target",
    "vmapping_get_state",
    "tee_console",
    "ArtifactManager",
    "build_human_play_layout",
    "build_human_play_setup_panel",
    "build_seed_setup_panel",
    "build_solution_path_panel",
    "build_vmapped_setup_panel",
    "save_solution_animation_and_frames",
    "PathStep",
    "build_path_steps_from_actions",
    "build_path_steps_from_nodes",
    "build_path_steps_from_trace",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "summaries": ("helpers.summaries", None),
    "plots": ("helpers.plots", None),
    "config_printer": ("helpers.config_printer", None),
    "print_config": ("helpers.config_printer", "print_config"),
    "human_format": ("helpers.formatting", "human_format"),
    "heuristic_dist_format": ("helpers.formatting", "heuristic_dist_format"),
    "qfunction_dist_format": ("helpers.formatting", "qfunction_dist_format"),
    "human_format_to_float": ("helpers.formatting", "human_format_to_float"),
    "img_to_colored_str": ("helpers.formatting", "img_to_colored_str"),
    "AimLogger": ("helpers.logger", "AimLogger"),
    "BaseLogger": ("helpers.logger", "BaseLogger"),
    "NoOpLogger": ("helpers.logger", "NoOpLogger"),
    "TensorboardLogger": ("helpers.logger", "TensorboardLogger"),
    "create_logger": ("helpers.logger", "create_logger"),
    "calculate_heuristic_metrics": ("helpers.metrics", "calculate_heuristic_metrics"),
    "RichProgressBar": ("helpers.rich_progress", "RichProgressBar"),
    "tqdm": ("helpers.rich_progress", "tqdm"),
    "trange": ("helpers.rich_progress", "trange"),
    "convert_to_serializable_dict": ("helpers.util", "convert_to_serializable_dict"),
    "flatten_dict": ("helpers.util", "flatten_dict"),
    "make_hashable": ("helpers.util", "make_hashable"),
    "display_value": ("helpers.util", "display_value"),
    "vmapping_search": ("helpers.search_utils", "vmapping_search"),
    "vmapping_init_target": ("helpers.search_utils", "vmapping_init_target"),
    "vmapping_get_state": ("helpers.search_utils", "vmapping_get_state"),
    "tee_console": ("helpers.capture", "tee_console"),
    "ArtifactManager": ("helpers.artifact_manager", "ArtifactManager"),
    "build_human_play_layout": ("helpers.visualization", "build_human_play_layout"),
    "build_human_play_setup_panel": (
        "helpers.visualization",
        "build_human_play_setup_panel",
    ),
    "build_seed_setup_panel": ("helpers.visualization", "build_seed_setup_panel"),
    "build_solution_path_panel": ("helpers.visualization", "build_solution_path_panel"),
    "build_vmapped_setup_panel": ("helpers.visualization", "build_vmapped_setup_panel"),
    "save_solution_animation_and_frames": (
        "helpers.visualization",
        "save_solution_animation_and_frames",
    ),
    "PathStep": ("helpers.path_steps", "PathStep"),
    "build_path_steps_from_actions": ("helpers.path_steps", "build_path_steps_from_actions"),
    "build_path_steps_from_nodes": ("helpers.path_steps", "build_path_steps_from_nodes"),
    "build_path_steps_from_trace": ("helpers.path_steps", "build_path_steps_from_trace"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

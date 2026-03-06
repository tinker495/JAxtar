import os
from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError


def iter_model_path_candidates(filename: str):
    """Yield plausible checkpoint path candidates in priority order."""
    seen: set[str] = set()
    candidates = [filename]
    path = Path(filename)
    if path.suffix == ".pkl" and path.stem.endswith("_v2"):
        candidates.append(str(path.with_name(f"{path.stem[:-3]}{path.suffix}")))

    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def resolve_model_path(filename: str) -> str:
    """Return the first locally available checkpoint path candidate."""
    for candidate in iter_model_path_candidates(filename):
        if os.path.exists(candidate):
            return candidate
    return filename


def is_model_downloaded(filename: str):
    return os.path.exists(resolve_model_path(filename))


def download_model(filename: str):
    last_error: EntryNotFoundError | None = None
    for candidate in iter_model_path_candidates(filename):
        try:
            huggingface_hub.hf_hub_download(
                repo_id="Tinker/JAxtar_models",
                repo_type="model",
                filename=candidate,
                local_dir="",
            )
            return resolve_model_path(candidate)
        except EntryNotFoundError as e:
            last_error = e

    if last_error is not None:
        raise FileNotFoundError(
            f"Checkpoint not found for any candidate path derived from '{filename}'."
        ) from last_error
    raise FileNotFoundError(f"Checkpoint not found: {filename}")

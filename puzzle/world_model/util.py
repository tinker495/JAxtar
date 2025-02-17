import os

import huggingface_hub


def is_dataset_downloaded():
    return os.path.exists("puzzle/world_model/data")


def download_dataset():
    huggingface_hub.snapshot_download(
        repo_id="Tinker/puzzle_world_model_ds",
        repo_type="dataset",
        local_dir="puzzle/world_model/",
    )

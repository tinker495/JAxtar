import os

import huggingface_hub


def is_world_model_dataset_downloaded():
    return os.path.exists("world_model_puzzle/data")


def download_world_model_dataset():
    huggingface_hub.snapshot_download(
        repo_id="Tinker/puzzle_world_model_ds",
        repo_type="dataset",
        local_dir="world_model_puzzle/",
    )

import os

import huggingface_hub


def is_model_downloaded(filename: str):
    return os.path.exists(filename)


def download_model(filename: str):
    huggingface_hub.hf_hub_download(
        repo_id="Tinker/JAxtar_models",
        repo_type="model",
        filename=filename,
        local_dir="",
    )

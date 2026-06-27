import importlib

import jax.numpy as jnp

from helpers.rich_progress import RichProgressBar, tqdm
from train_util.target_update import soft_update


def test_progress_alias_tqdm_class_is_removed():
    rich_progress = importlib.import_module("helpers.rich_progress")
    assert not hasattr(rich_progress, "tqdm_class")
    assert isinstance(tqdm([]), RichProgressBar)


def test_optional_batch_stats_alias_is_removed():
    train_util = importlib.import_module("train_util.util")
    assert hasattr(train_util, "apply_with_conditional_batch_stats")
    assert not hasattr(train_util, "apply_with_optional_batch_stats")


def test_soft_update_remains_importable_with_legacy_weighting():
    target_params = {"w": jnp.array([10.0, 20.0])}
    params = {"w": jnp.array([2.0, 4.0])}

    updated = soft_update(target_params, params, 0.25)

    assert jnp.allclose(updated["w"], jnp.array([4.0, 8.0]))

import jax.numpy as jnp

from helpers.rich_progress import RichProgressBar, tqdm_class
from train_util.target_update import soft_update


def test_tqdm_class_alias_remains_importable():
    assert tqdm_class is RichProgressBar


def test_soft_update_remains_importable_with_legacy_weighting():
    target_params = {"w": jnp.array([10.0, 20.0])}
    params = {"w": jnp.array([2.0, 4.0])}

    updated = soft_update(target_params, params, 0.25)

    assert jnp.allclose(updated["w"], jnp.array([4.0, 8.0]))

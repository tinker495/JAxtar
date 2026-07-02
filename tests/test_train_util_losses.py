import jax.numpy as jnp
import numpy as np
import pytest

from train_util.losses import loss_from_diff


def test_loss_from_diff_mse_squares_diff_directly():
    diff = jnp.array([-2.0, 0.5, 3.0])

    np.testing.assert_allclose(loss_from_diff(diff, loss="mse"), jnp.array([4.0, 0.25, 9.0]))


def test_loss_from_diff_rejects_invalid_loss_name():
    with pytest.raises(ValueError, match="Unsupported loss"):
        loss_from_diff(jnp.array([1.0]), loss="not-a-loss")  # type: ignore[arg-type]

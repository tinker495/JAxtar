from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from train_util.losses import loss_from_diff


class DistanceModel(ABC, nn.Module):
    action_size: int = 1

    @abstractmethod
    def __call__(self, x, training=False):
        pass

    def train_loss(self, x, target, actions=None, loss_type="mse", loss_args=None, **kwargs):
        pred = self(x, training=True)
        if actions is not None:
            pred = jnp.take_along_axis(pred, actions[:, jnp.newaxis], axis=1)

        diff = target - pred
        return loss_from_diff(diff, loss=loss_type, loss_args=loss_args)


class DistanceHLGModel(ABC, nn.Module):
    action_size: int = 1

    categorial_n: int = 100
    vmin: float = -1.0
    vmax: float = 30.0
    _sigma: float = 0.75

    @abstractmethod
    def __call__(self, x, training=False):
        pass

    def train_loss(self, x, target, actions=None, **kwargs):
        categorial_bins, sigma = self.categorial_bins, self.sigma
        # target: [batch, 1]

        def f(target):
            cdf_evals = jax.scipy.special.erf((categorial_bins - target) / (jnp.sqrt(2) * sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(f)(target)
        logits_actions = self.get_logits(x, training=True)
        if actions is None:
            logits = logits_actions.squeeze(1)
        else:
            logits = jnp.take_along_axis(
                logits_actions, actions[:, jnp.newaxis, jnp.newaxis], axis=1
            ).squeeze(1)
        sce = optax.softmax_cross_entropy(logits, target_probs)  # (batch_size,)
        return sce

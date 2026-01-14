from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from train_util.losses import loss_from_diff
from train_util.train_logs import TrainLogInfo


class DistanceModel(ABC, nn.Module):
    action_size: int = 1

    @abstractmethod
    def __call__(self, x, training=False):
        pass

    def train_loss(self, x, target, actions=None, loss_type="mse", loss_args=None, **kwargs):
        pred = self(x, training=True)
        if actions is not None:
            pred = jnp.take_along_axis(pred, actions, axis=1)
        else:
            pred = pred.squeeze(1)

        diff = target - pred
        loss = loss_from_diff(diff, loss=loss_type, loss_args=loss_args)
        log_infos = {
            "Metrics/pred": TrainLogInfo(pred),
            "Losses/diff": TrainLogInfo(diff, log_mean=False),
            "Losses/abs_diff": TrainLogInfo(jnp.abs(diff), log_mean=True, log_histogram=False),
            "Losses/mse": TrainLogInfo(jnp.mean(diff**2), log_histogram=False),
            "Losses/loss": TrainLogInfo(loss, log_histogram=False),
        }
        return loss, log_infos


class DistanceHLGModel(ABC, nn.Module):
    action_size: int = 1

    categorial_n: int = 100
    vmin: float = -1.0
    vmax: float = 30.0
    _sigma: float = 0.75

    def setup(self):
        self.categorial_bins = np.linspace(
            self.vmin, self.vmax, self.categorial_n + 1
        )  # (categorial_n + 1,)
        categorial_centers = (
            self.categorial_bins[:-1] + self.categorial_bins[1:]
        ) / 2  # (categorial_n,)
        self.categorial_centers = categorial_centers.reshape(1, 1, -1)  # (1, 1, categorial_n)
        self.sigma = self._sigma * (self.categorial_bins[1] - self.categorial_bins[0])

    @abstractmethod
    def get_logits(self, x, training=False):
        pass

    def logit_to_values(self, logits):
        softmax = jax.nn.softmax(logits, axis=-1)
        categorial_centers = self.categorial_centers
        x = jnp.sum(softmax * categorial_centers, axis=-1)  # (batch_size, action_size)
        return x

    def __call__(self, x, training=False):
        logits = self.get_logits(x, training)
        return self.logit_to_values(logits)

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

        pred_actions = self.logit_to_values(logits_actions)

        if actions is None:
            logits = logits_actions.squeeze(1)
            pred = pred_actions.squeeze(1)
        else:
            logits = jnp.take_along_axis(logits_actions, actions[:, jnp.newaxis], axis=1).squeeze(1)
            pred = jnp.take_along_axis(pred_actions, actions[:, jnp.newaxis], axis=1).squeeze(1)

        sce = optax.softmax_cross_entropy(logits, target_probs)  # (batch_size,)
        log_infos = {
            "Metrics/pred": TrainLogInfo(pred),
            "Losses/sce": TrainLogInfo(sce, log_histogram=False),
            "Losses/loss": TrainLogInfo(sce, log_histogram=False),
            "Losses/diff": TrainLogInfo(target - pred, log_mean=False),
            "Losses/abs_diff": TrainLogInfo(
                jnp.abs(target - pred), log_mean=True, log_histogram=False
            ),
        }
        return sce, log_infos

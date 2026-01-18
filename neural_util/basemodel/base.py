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
        log_infos = (
            TrainLogInfo("Metrics/pred", pred),
            TrainLogInfo("Losses/diff", diff, log_mean=False),
            TrainLogInfo("Losses/mae", jnp.abs(diff), log_histogram=False),
            TrainLogInfo("Losses/mse", jnp.mean(diff**2), log_histogram=False),
            TrainLogInfo("Losses/loss", loss, log_histogram=False),
        )
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
        diff = target - pred
        log_infos = (
            TrainLogInfo("Metrics/pred", pred),
            TrainLogInfo("Losses/sce", sce, log_histogram=False),
            TrainLogInfo("Losses/loss", sce, log_histogram=False),
            TrainLogInfo("Losses/diff", diff, log_mean=False),
            TrainLogInfo("Losses/mae", jnp.abs(diff), log_histogram=False),
            TrainLogInfo("Losses/mse", jnp.mean(diff**2), log_histogram=False),
        )
        return sce, log_infos


class DistanceGroupDIRModel(ABC, nn.Module):
    action_size: int = 1

    categorial_n: int = 10
    vmin: float = -1.0
    vmax: float = 26.0
    _sigma: float = 1.5

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
    def get_logits_and_moe_values(self, x, training=False) -> tuple[jnp.ndarray, jnp.ndarray]:
        # (batch_size, action_size, categorial_n), (batch_size, action_size, categorial_n)
        pass

    def logit_and_moe_values_to_values(self, logits, moe_values):
        softmax = jax.nn.softmax(logits, axis=-1)  # (batch_size, action_size, categorial_n)
        x = jnp.sum(softmax * moe_values, axis=-1)  # (batch_size, action_size)
        return x  # (batch_size, action_size)

    def __call__(self, x, training=False):
        logits, moe_values = self.get_logits_and_moe_values(
            x, training
        )  # (batch_size, action_size, categorial_n), (batch_size, action_size, categorial_n)
        return self.logit_and_moe_values_to_values(logits, moe_values)  # (batch_size, action_size)

    def train_loss(self, x, target, actions=None, loss_type="mse", loss_args=None, **kwargs):
        categorial_bins, sigma = self.categorial_bins, self.sigma
        # target: [batch, 1]

        def f(target):
            cdf_evals = jax.scipy.special.erf((categorial_bins - target) / (jnp.sqrt(2) * sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(f)(target)
        logits_actions, moe_values_actions = self.get_logits_and_moe_values(
            x, training=True
        )  # (batch_size, action_size, categorial_n), (batch_size, action_size, categorial_n)

        pred_actions = self.logit_and_moe_values_to_values(logits_actions, moe_values_actions)

        if actions is None:
            logits = logits_actions.squeeze(1)  # (batch_size, categorial_n)
            moe_values = moe_values_actions.squeeze(1)  # (batch_size, categorial_n)
            pred = pred_actions.squeeze(1)  # (batch_size,)
        else:
            logits = jnp.take_along_axis(logits_actions, actions[:, jnp.newaxis], axis=1).squeeze(
                1
            )  # (batch_size, categorial_n)
            moe_values = jnp.take_along_axis(
                moe_values_actions, actions[:, jnp.newaxis], axis=1
            ).squeeze(
                1
            )  # (batch_size, categorial_n)
            pred = jnp.take_along_axis(pred_actions, actions[:, jnp.newaxis], axis=1).squeeze(
                1
            )  # (batch_size,)

        softmax = jax.nn.softmax(logits, axis=-1)  # (batch_size, categorial_n)
        sce = optax.softmax_cross_entropy(logits, target_probs)  # (batch_size,)
        diffes = target[:, jnp.newaxis] - moe_values  # (batch_size, categorial_n)
        moe_regress_losses = loss_from_diff(
            diffes, loss=loss_type, loss_args=loss_args
        )  # (batch_size, categorial_n)
        regress_losses = jnp.sum(softmax * moe_regress_losses, axis=-1)  # (batch_size,)

        regression_gain = kwargs.get("regression_gain", 0.1)
        total_loss = sce + regression_gain * regress_losses  # (batch_size,)
        diff = target - pred  # (batch_size,)
        log_infos = (
            TrainLogInfo("Metrics/pred", pred),
            TrainLogInfo("Losses/sce", sce, log_histogram=False),
            TrainLogInfo("Losses/regress", regress_losses, log_histogram=False),
            TrainLogInfo("Losses/loss", total_loss, log_histogram=False),
            TrainLogInfo("Losses/diff", diff, log_mean=False),
            TrainLogInfo("Losses/mae", jnp.abs(diff), log_histogram=False),
            TrainLogInfo("Losses/mse", jnp.mean(diff**2), log_histogram=False),
        )
        return total_loss, log_infos

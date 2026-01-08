from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from train_util.losses import loss_from_diff
from train_util.train_logs import TrainLogInfo


class SelfPredictiveMixin(ABC, nn.Module):
    spr_loss_scale: float = 1.0

    @abstractmethod
    def states_to_latents(self, x, training=False):
        """
        Convert states to latents.
        This is used to convert the states to the latent space of the model.
        """
        pass

    @abstractmethod
    def latents_to_projection(self, x, training=False):
        """
        Convert latents to projection.
        This is used to convert the latents to the state space of the model.
        """
        pass

    @abstractmethod
    def latents_to_distances(self, x, training=False):
        """
        Convert latents to distances.
        This is used to convert the latents to the distance space of the model.
        """
        pass

    @abstractmethod
    def predict_ema_latents(self, x, training=False):
        """
        Predict ema latents.
        This is used to predict the ema latents given the current latents.
        """
        pass

    @abstractmethod
    def transition(self, latents, actions, training=False):
        """
        Return the transition of the latents.
        The transition is the transition of the latents given actions.
        """
        pass

    def initialize_components(self, x, training=False):
        latents = self.states_to_latents(x, training=training)  # [..., latent_dim]
        dists = self.latents_to_distances(latents, training=training)  # [..., action_size]

        # Initialize transition model
        batch_size = latents.shape[0]
        # Ensure dummy actions are int32 and shape (Batch, 1) for compatibility
        dummy_actions = jnp.zeros((batch_size), dtype=jnp.int32)
        _ = self.transition(latents, dummy_actions, training=training)

        # Initialize projection and prediction heads
        proj = self.latents_to_projection(latents, training=training)
        _ = self.predict_ema_latents(proj, training=training)

        return dists

    def __call__(self, x, training=False):
        latents = self.states_to_latents(x, training=training)  # [..., latent_dim]
        dists = self.latents_to_distances(latents, training=training)  # [..., action_size]
        return dists

    def compute_self_predictive_loss(
        self, start_latents, path_actions, ema_latents, same_trajectory_masks, training=True
    ):
        if ema_latents is None:
            return 0.0

        # latents: (Batch, Time, Dim)
        # path_actions: (Batch, Time - 1)
        # ema_latents: (Batch, Time - 1, Dim)
        # same_trajectory_masks: (Batch, Time - 1)

        transition_actions = jnp.swapaxes(path_actions, 0, 1)  # (Time - 1, Batch)

        def body(current_latents, action):
            # Predict next latent: z_{t+1} = Trans(z_t, a_t)
            next_latents = self.transition(current_latents, action, training=training)
            return next_latents, next_latents

        _, next_latents = jax.lax.scan(body, start_latents, transition_actions)
        next_projections = self.latents_to_projection(
            next_latents, training=training
        )  # (Time - 1, Batch, ProjectionDim)
        next_predictions = self.predict_ema_latents(
            next_projections, training=training
        )  # (Time - 1, Batch, ProjectionDim)

        next_predictions = jnp.swapaxes(next_predictions, 0, 1)  # (Batch, Time - 1, ProjectionDim)

        cosine_similarity = optax.cosine_similarity(
            next_predictions, ema_latents
        )  # (Batch, Time - 1)
        loss = 1.0 - cosine_similarity  # (Batch, Time - 1)
        if same_trajectory_masks is not None:
            loss = loss * same_trajectory_masks

        loss = jnp.sum(loss, axis=-1, keepdims=True)  # Sum over time, resulting in (Batch, 1)
        return self.spr_loss_scale * loss


class SelfPredictiveDistanceModel(SelfPredictiveMixin):
    action_size: int = 1

    def train_loss(
        self,
        x,
        target,
        actions=None,
        loss_type="mse",
        loss_args=None,
        path_actions=None,
        ema_latents=None,
        same_trajectory_masks=None,
        **kwargs
    ):
        latents = self.states_to_latents(x, training=True)  # [..., latent_dim]

        dists = self.latents_to_distances(latents, training=True)  # [..., action_size]
        if actions is not None:
            dists = jnp.take_along_axis(dists, actions, axis=-1)  # [..., 1]
        else:
            dists = dists.squeeze(-1)  # [...,]

        diff = target - dists  # [..., ]
        dist_loss = loss_from_diff(diff, loss=loss_type, loss_args=loss_args)  # [...,]

        spr_loss = self.compute_self_predictive_loss(
            latents[:, 0], path_actions, ema_latents, same_trajectory_masks, training=True
        )  # (Batch, 1)
        total_loss = dist_loss + spr_loss
        log_infos = {
            "Metrics/pred": TrainLogInfo(dists),
            "Losses/diff": TrainLogInfo(diff, log_mean=False),
            "Losses/loss": TrainLogInfo(total_loss, log_histogram=False),
            "Losses/dist_loss": TrainLogInfo(dist_loss, log_histogram=False),
            "Losses/spr_loss": TrainLogInfo(spr_loss, log_histogram=False),
        }
        return total_loss, log_infos


class SelfPredictiveDistanceHLGModel(SelfPredictiveMixin):
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
    def latents_to_logits(self, x, training=False):
        """
        Convert latents to logits.
        This is used to convert the latents to the logits of the model.
        """
        pass

    def latents_to_distances(self, latents, training=False):
        logits = self.latents_to_logits(latents, training)  # [..., action_size, categorial_n]
        softmax = jax.nn.softmax(logits, axis=-1)
        categorial_centers = self.categorial_centers  # (1, 1, categorial_n)
        x = jnp.sum(softmax * categorial_centers, axis=-1)  # [..., action_size]
        return x

    def train_loss(
        self,
        x,
        target,
        actions=None,
        path_actions=None,
        ema_latents=None,
        same_trajectory_masks=None,
        **kwargs
    ):
        categorial_bins, sigma = self.categorial_bins, self.sigma
        # target: [batch, 1]

        def f(target):
            cdf_evals = jax.scipy.special.erf((categorial_bins - target) / (jnp.sqrt(2) * sigma))
            z = cdf_evals[-1] - cdf_evals[0]
            bin_probs = cdf_evals[1:] - cdf_evals[:-1]
            return bin_probs / z

        target_probs = jax.vmap(jax.vmap(f))(target)  # [..., action_size, categorial_n]

        latents = self.states_to_latents(x, training=True)  # [..., latent_dim]

        logits_actions = self.latents_to_logits(
            latents, training=True
        )  # [..., action_size, categorial_n]
        pred_actions = self.latents_to_values(logits_actions)

        if actions is None:
            logits = logits_actions.squeeze(-2)  # [..., categorial_n]
            pred = pred_actions.squeeze(-1)
        else:
            logits = jnp.take_along_axis(
                logits_actions, actions[..., jnp.newaxis, jnp.newaxis], axis=-2
            ).squeeze(
                -2
            )  # [..., categorial_n]
            pred = jnp.take_along_axis(pred_actions, actions[..., jnp.newaxis], axis=-1).squeeze(-1)

        dist_loss = optax.softmax_cross_entropy(logits, target_probs)  # [..., ]
        dist_loss = jnp.nan_to_num(dist_loss, nan=0.0, posinf=1e6, neginf=-1e6)

        # Keep original dist_loss for sce logging before any potential mean
        sce = dist_loss

        if dist_loss.ndim > 1:
            dist_loss = jnp.mean(dist_loss, axis=-1)

        if latents.ndim >= 3:
            # We pass full latents to SPR loss for sequence handling
            spr_latents = latents
        else:
            spr_latents = latents

        spr_loss = self.compute_self_predictive_loss(
            spr_latents, path_actions, ema_latents, same_trajectory_masks, training=True
        )  # [...,]
        spr_loss = jnp.nan_to_num(spr_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        total_loss = dist_loss + spr_loss
        log_infos = {
            "Metrics/pred": TrainLogInfo(pred),
            "Losses/diff": TrainLogInfo(target - pred, log_mean=False),
            "Losses/loss": TrainLogInfo(total_loss, log_histogram=False),
            "Losses/dist_loss": TrainLogInfo(dist_loss, log_histogram=False),
            "Losses/spr_loss": TrainLogInfo(spr_loss, log_histogram=False),
            "Losses/sce": TrainLogInfo(sce, log_histogram=False),
        }
        return total_loss, log_infos

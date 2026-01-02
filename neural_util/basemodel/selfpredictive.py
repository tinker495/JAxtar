from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from train_util.losses import loss_from_diff


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
        action_size = getattr(self, "action_size", 1)
        dummy_actions = jnp.zeros((batch_size, action_size), dtype=latents.dtype)
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
        self, last_latents, path_actions, ema_latents, same_trajectory_masks, training=True
    ):
        if ema_latents is None:
            return 0.0

        # Swap axes to make time axis 0 for scan
        # path_actions: (..., Time) -> (Time, ...)
        path_actions = jnp.moveaxis(path_actions, -1, 0)
        # ema_latents: (..., Time, Dim) -> (Time, ..., Dim)
        ema_latents = jnp.moveaxis(ema_latents, -2, 0)
        if same_trajectory_masks is not None:
            same_trajectory_masks = jnp.moveaxis(same_trajectory_masks, -1, 0)
        last_latents = jnp.float32(last_latents)

        def body(last_latents, path_actions):
            # path_actions from scan is (Batch, 1) or (1024, 1)
            # path_actions need to be expanded to match (Batch, 1) for concatenation if needed,
            # but usually it's already (Batch,) or (Batch, 1)
            # here path_actions comes from scan over (Time, Batch), so it is (Batch,)
            path_actions = path_actions[..., jnp.newaxis]  # (Batch, 1)
            predicted_next_latents = self.transition(last_latents, path_actions, training=training)
            # Ensure the output has the same dtype as the input carry (last_latents)
            predicted_next_latents = predicted_next_latents.astype(last_latents.dtype)
            return predicted_next_latents, predicted_next_latents

        _, predicted_next_latents = jax.lax.scan(body, last_latents, path_actions)

        # Restore axes: (Time, ..., Dim) -> (..., Time, Dim)
        predicted_next_latents = jnp.moveaxis(predicted_next_latents, 0, -2)
        ema_latents = jnp.moveaxis(ema_latents, 0, -2)
        if same_trajectory_masks is not None:
            same_trajectory_masks = jnp.moveaxis(same_trajectory_masks, 0, -1)

        projected_next_latents = self.latents_to_projection(
            predicted_next_latents, training=training
        )
        predicted_next_latents = self.predict_ema_latents(projected_next_latents, training=training)

        # Truncate predicted_next_latents or ema_latents if sizes differ
        time_axis = -2
        min_len = min(predicted_next_latents.shape[time_axis], ema_latents.shape[time_axis])

        predicted_next_latents = predicted_next_latents[..., :min_len, :]
        ema_latents = ema_latents[..., :min_len, :]
        if same_trajectory_masks is not None:
            same_trajectory_masks = same_trajectory_masks[..., :min_len]

        cosine_similarity = optax.cosine_similarity(predicted_next_latents, ema_latents)
        loss = 1.0 - cosine_similarity

        if same_trajectory_masks is not None:
            loss = loss * same_trajectory_masks

        # Mean over time axis (last axis of cosine_similarity result)
        loss = jnp.sum(loss, axis=-1)

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
        dist_loss = jnp.nan_to_num(dist_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        if dist_loss.ndim > 1:
            dist_loss = jnp.mean(dist_loss, axis=-1)

        if latents.ndim >= 3:
            last_latents = latents[..., 0, :]
        else:
            last_latents = latents
        spr_loss = self.compute_self_predictive_loss(
            last_latents, path_actions, ema_latents, same_trajectory_masks, training=True
        )  # [...,]
        spr_loss = jnp.nan_to_num(spr_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        return dist_loss + spr_loss  # [...,]


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

        if actions is None:
            logits = logits_actions.squeeze(-2)  # [..., categorial_n]
        else:
            logits = jnp.take_along_axis(
                logits_actions, actions[..., jnp.newaxis], axis=-2
            ).squeeze()  # [..., categorial_n]
        dist_loss = optax.softmax_cross_entropy(logits, target_probs)  # [..., ]
        dist_loss = jnp.nan_to_num(dist_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        if dist_loss.ndim > 1:
            dist_loss = jnp.mean(dist_loss, axis=-1)

        if latents.ndim >= 3:
            last_latents = latents[..., 0, :]
        else:
            last_latents = latents
        spr_loss = self.compute_self_predictive_loss(
            last_latents, path_actions, ema_latents, same_trajectory_masks, training=True
        )  # [...,]
        spr_loss = jnp.nan_to_num(spr_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        return dist_loss + spr_loss  # [...,]

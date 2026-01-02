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
        # Ensure dummy actions are int32 and shape (Batch, 1) for compatibility
        dummy_actions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
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
        self, latents, path_actions, ema_latents, same_trajectory_masks, training=True
    ):
        if ema_latents is None:
            return 0.0

        # latents: (Batch, Time, Dim)
        # path_actions: (Batch, Time) (or Time, Batch if transposed before?)
        # We assume (Batch, Time) coming from the refactored training loop.

        # Slicing:
        # Start state z_0
        start_latents = latents[:, 0]  # (Batch, Dim)

        # Actions a_0 ... a_{K-2} (actions to transition z_0->z_1 ... z_{K-2}->z_{K-1})
        # path_actions corresponds to actions taken at state t.
        # So we want path_actions[:, 0:-1]
        transition_actions = path_actions[:, :-1]  # (Batch, Time-1)

        # Targets are ema_latents (projections of z_1 ... z_{K-1})
        # passed in already sliced/computed by get_self_predictive_train_args
        # ema_latents: (Batch, Time-1, Dim)

        # Move Time to axis 0 for scan: (Time-1, Batch, ...)
        transition_actions = jnp.swapaxes(transition_actions, 0, 1)
        target_projections = jnp.swapaxes(ema_latents, 0, 1)

        if same_trajectory_masks is not None:
            # (Batch, Time-1) -> (Time-1, Batch)
            same_trajectory_masks = jnp.swapaxes(same_trajectory_masks, 0, 1)

        start_latents = jnp.float32(start_latents)

        def body(current_latents, inputs):
            action, target_proj, mask = inputs

            # action: (Batch,)
            action = action[..., jnp.newaxis]  # (Batch, 1)

            # Predict next latent: z_{t+1} = Trans(z_t, a_t)
            next_latents = self.transition(current_latents, action, training=training)
            next_latents = next_latents.astype(current_latents.dtype)

            # Predict projection: p_{t+1} = Pred(Proj(z_{t+1}))
            # Wait, standard SPR is:
            # y = Proj(z_{t+1})
            # pred = Predictor(y)
            # loss = - cosine(pred, target_ema_proj)

            # In existing code:
            # projected_next_latents = self.latents_to_projection(predicted_next_latents)
            # predicted_next_latents = self.predict_ema_latents(projected_next_latents)

            current_projection = self.latents_to_projection(next_latents, training=training)
            prediction = self.predict_ema_latents(current_projection, training=training)

            cosine_similarity = optax.cosine_similarity(prediction, target_proj)
            step_loss = 1.0 - cosine_similarity

            if mask is not None:
                step_loss = step_loss * mask

            return next_latents, step_loss

        # Scan over time
        # We need to bundle inputs for scan
        # If same_trajectory_masks is None, we need to handle that.
        if same_trajectory_masks is None:
            # Create dummy mask of 1s or handle logic.
            # Easier to just use None in tree_map if scan supports it,
            # but explicit is better.
            same_trajectory_masks = jnp.ones(transition_actions.shape[:2], dtype=jnp.float32)

        _, losses = jax.lax.scan(
            body, start_latents, (transition_actions, target_projections, same_trajectory_masks)
        )

        # losses is (Time-1, Batch)
        loss = jnp.sum(losses, axis=0)  # Sum over time, resulting in (Batch,)

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
            # We pass full latents to SPR loss for sequence handling
            spr_latents = latents
        else:
            spr_latents = latents

        spr_loss = self.compute_self_predictive_loss(
            spr_latents, path_actions, ema_latents, same_trajectory_masks, training=True
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
            # We pass full latents to SPR loss for sequence handling
            spr_latents = latents
        else:
            spr_latents = latents

        spr_loss = self.compute_self_predictive_loss(
            spr_latents, path_actions, ema_latents, same_trajectory_masks, training=True
        )  # [...,]
        spr_loss = jnp.nan_to_num(spr_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        return dist_loss + spr_loss  # [...,]

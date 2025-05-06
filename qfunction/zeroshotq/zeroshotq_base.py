import os
import pickle
from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from neural_util.modules import DTYPE, LayerNorm, SimbaResBlock
from neural_util.util import download_model, is_model_downloaded
from puzzle.puzzle_base import Puzzle
from qfunction.q_base import QFunction


class Projector(nn.Module):
    latent_dim: int = 256
    Res_N: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latent_dim, dtype=DTYPE)(x)
        for _ in range(self.Res_N):
            x = SimbaResBlock(self.latent_dim * 4, LayerNorm)(x)
        x = LayerNorm(x)
        x = nn.Dense(self.latent_dim, dtype=DTYPE)(x)
        return x


class FixedSolveConfigProjector(nn.Module):
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        latent = self.param(
            "solve_config_latent", nn.initializers.normal(stddev=0.02), (1, self.latent_dim)
        )  # (1, latent_dim)
        output = jnp.tile(latent, (batch_size, 1))  # (batch_size, latent_dim)
        return output


class GoalProjector(nn.Module):
    action_size: int = 4
    latent_dim: int = 256
    Res_N: int = 2
    fixed_target: bool = False

    def setup(self):
        self.backward_projector = Projector(
            latent_dim=self.latent_dim * self.action_size, Res_N=self.Res_N
        )
        if self.fixed_target:
            self.solve_config_projector = FixedSolveConfigProjector(latent_dim=self.latent_dim)
        else:
            self.solve_config_projector = Projector(latent_dim=self.latent_dim, Res_N=self.Res_N)

    def __call__(self, solve_config, current):
        z = self.solve_config_projection(solve_config)  # (batch_size, latent_dim)
        b_a = self.backward_projection(current)  # (batch_size, action_size, latent_dim)
        return z, b_a

    def solve_config_projection(self, solve_config):
        z = self.solve_config_projector(solve_config)  # (batch_size, latent_dim)
        return z

    def backward_projection(self, state):
        b_a = self.backward_projector(state)  # (batch_size, latent_dim * action_size)
        b_a = jnp.reshape(
            b_a, (b_a.shape[0], self.action_size, self.latent_dim)
        )  # (batch_size, action_size, latent_dim)
        return b_a

    def get_b(
        self,
        state: Puzzle.State,
        actions: chex.Array,
    ):
        b_a = self.backward_projection(state)  # (batch_size, action_size, latent_dim)
        b = jnp.take_along_axis(
            b_a, actions[:, jnp.newaxis, jnp.newaxis], axis=1
        )  # (batch_size, 1, latent_dim)
        b = jnp.reshape(b, (b.shape[0], self.latent_dim))  # (batch_size, latent_dim)
        return b  # (batch_size, latent_dim)


class ZeroshotQModelBase(nn.Module):
    action_size: int = 4
    latent_dim: int = 256
    Res_N: int = 2

    def setup(self):

        self.forward_projector = Projector(
            latent_dim=self.latent_dim * self.action_size, Res_N=self.Res_N
        )

        self.forward_weight = self.param("forward_weight", nn.initializers.normal(stddev=1), (1,))
        self.forward_bias = self.param("forward_bias", nn.initializers.normal(stddev=1), (1,))

    def __call__(self, state, z):
        f_a = self.forward_projection(state, z)  # (batch_size, action_size, latent_dim)
        q = self.distance(f_a, z)  # (batch_size, action_size)
        return q

    def distance(self, f_a, b):  # (batch_size, action_size, latent_dim), (batch_size, latent_dim)
        f_a = jnp.einsum("baz,bz->ba", f_a, b)  # (batch_size, action_size)
        f_a = f_a * self.forward_weight + self.forward_bias
        q = -jax.nn.log_sigmoid(f_a)
        return q

    def forward_projection(self, state, z):
        f_a = self.forward_projector(jnp.concatenate([state, z], axis=-1))
        f_a = jnp.reshape(
            f_a, (f_a.shape[0], self.action_size, self.latent_dim)
        )  # (batch_size, action_size, latent_dim)
        return f_a


class ZeroshotQFunctionBase(QFunction):
    def __init__(
        self,
        puzzle: Puzzle,
        zeroshot_model: nn.Module = ZeroshotQModelBase,
        goal_projector: nn.Module = GoalProjector,
        init_params: bool = True,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.is_fixed = puzzle.fixed_target
        self.action_size = self._get_action_size()
        self.model = zeroshot_model(self.action_size, **kwargs)
        self.goal_projector = goal_projector(self.action_size, fixed_target=self.is_fixed, **kwargs)
        if init_params:
            self.params, self.goal_params = self.get_new_params()

    def _get_action_size(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        return self.puzzle.get_neighbours(dummy_solve_config, dummy_current)[0].shape[0][0]

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        dummy_solve_config_processed = self.pre_process_solve_config(dummy_solve_config)
        dummy_current_processed = self.pre_process_state(dummy_current)

        goal_params = self.goal_projector.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(dummy_current_processed, axis=0),
            jnp.expand_dims(dummy_solve_config_processed, axis=0),
        )

        model_params = self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            jnp.expand_dims(dummy_current_processed, axis=0),
            jnp.zeros((1, self.model.latent_dim)),
        )
        return model_params, goal_params

    @classmethod
    def load_model(cls, puzzle: Puzzle, path: str):

        try:
            if not is_model_downloaded(path):
                download_model(path)
            with open(path, "rb") as f:
                params = pickle.load(f)
            qfunc = cls(puzzle, init_params=False)
            dummy_solve_config = puzzle.SolveConfig.default()
            dummy_current = puzzle.State.default()
            qfunc.model.apply(
                params,
                jnp.expand_dims(qfunc.pre_process(dummy_solve_config, dummy_current), axis=0),
                training=False,
            )  # check if the params are compatible with the model
            qfunc.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            qfunc = cls(puzzle)
        return qfunc

    def save_model(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def batched_q_value(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        return self.batched_param_q_value(self.params, solve_config, current)

    def batched_param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        # Preprocess inputs
        # Use vmap for solve_config as it might be a single instance repeated or a batch
        solve_config_processed = jax.vmap(self.pre_process_solve_config)(solve_config)
        current_processed = jax.vmap(self.pre_process_state)(current)

        # Get z using solve_config_projection method
        # Pass only necessary params to apply (typically the 'params' dict)
        z = self.goal_projector.apply(
            params,  # Pass the full param tree
            solve_config_processed,  # Input for solve_config_projection
            training=False,
            mutable=False,  # Don't need batch_stats or mutable state here
            method=self.goal_projector.solve_config_projection,
        )

        # Apply main model to get q_values
        # Assume model.apply might update batch_stats if present in params
        q_values, _ = self.model.apply(
            params,  # Pass the full param tree
            current_processed,
            z,  # Pass state and z
            training=False,
            mutable=["batch_stats"],
        )
        # q_values = self.post_process(q_values) # Post-processing happens inside model.apply/distance now
        return q_values

    def q_value(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
        """
        This function should return the distance between the state and the target.
        """
        return self.param_q_value(self.params, solve_config, current)[0]

    def param_q_value(
        self, params, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        # Preprocess inputs
        solve_config_processed = self.pre_process_solve_config(solve_config)
        current_processed = self.pre_process_state(current)
        # Add batch dimension
        solve_config_processed_b = jnp.expand_dims(solve_config_processed, axis=0)
        current_processed_b = jnp.expand_dims(current_processed, axis=0)

        # Get z using solve_config_projection method
        z = self.goal_projector.apply(
            params,  # Pass the full param tree
            solve_config_processed_b,  # Input for solve_config_projection
            training=False,
            mutable=False,  # Don't need batch_stats or mutable state here
            method=self.goal_projector.solve_config_projection,
        )

        # Apply main model
        # Assume model.apply might update batch_stats if present in params
        q_values, _ = self.model.apply(
            params,  # Pass the full param tree
            current_processed_b,
            z,  # Pass state and z
            training=False,
            mutable=["batch_stats"],
        )
        # q_values = self.post_process(q_values) # Post-processing happens inside model.apply/distance now
        return q_values[0]  # Return single value, remove batch dim

    def pre_process_solve_config(self, solve_config: Puzzle.SolveConfig) -> chex.Array:
        """
        This function should return the pre-processed solve_config.
        """
        if self.is_fixed:
            return jnp.zeros((1,))
        if self.puzzle.only_target:
            return self.pre_process_state(solve_config.TargetState)
        else:
            assert False, "This function should be implemented in the subclass"

    @abstractmethod
    def pre_process_state(self, state: Puzzle.State) -> chex.Array:
        """
        This function should return the pre-processed state.
        """
        pass

    def post_process(self, x: chex.Array) -> float:
        """
        This function should return the post-processed distance.
        """
        return x

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


class ActionProjector(nn.Module):
    action_size: int = 4
    latent_dim: int = 256
    Res_N: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latent_dim, dtype=DTYPE)(x)
        for _ in range(self.Res_N):
            x = SimbaResBlock(self.latent_dim * 4, LayerNorm)(x)
        x = LayerNorm(x)
        x = nn.Dense(self.action_size * self.latent_dim, dtype=DTYPE)(x)
        x = jnp.reshape(x, (x.shape[0], self.action_size, self.latent_dim))
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


class ZeroshotQModelBase(nn.Module):
    action_size: int = 4
    latent_dim: int = 256
    Res_N: int = 2
    fixed_target: bool = False

    def setup(self):
        self.forward_projector = ActionProjector(
            action_size=self.action_size, latent_dim=self.latent_dim, Res_N=self.Res_N
        )
        self.backward_projector = Projector(latent_dim=self.latent_dim, Res_N=self.Res_N)

        if self.fixed_target:
            self.solve_config_projector = FixedSolveConfigProjector(latent_dim=self.latent_dim)
        else:
            self.solve_config_projector = Projector(latent_dim=self.latent_dim, Res_N=self.Res_N)

        self.forward_weight = self.param("forward_weight", nn.initializers.normal(stddev=1), (1,))
        self.forward_bias = self.param("forward_bias", nn.initializers.normal(stddev=1), (1,))

    def __call__(self, solve_config, state):
        z = self.solve_config_projection(solve_config)  # (batch_size, latent_dim)
        bz = self.backward_projection(state)  # (batch_size, latent_dim)
        f_a = self.forward_projection(state, z)  # (batch_size, action_size, latent_dim)
        q = self.distance(f_a, bz)  # (batch_size, action_size)
        return q

    def solve_config_projection(self, solve_config):
        z = self.solve_config_projector(solve_config)  # (batch_size, latent_dim)
        return z

    def forward_projection(self, state, z):
        f_a = self.forward_projector(
            jnp.concatenate([state, z], axis=-1)
        )  # (batch_size, action_size, latent_dim)
        return f_a

    def backward_projection(self, state):
        bz = self.backward_projector(state)  # (batch_size, latent_dim)
        return bz

    def distance(self, f_a, z):  # (batch_size, action_size, latent_dim), (batch_size, latent_dim)
        f_a = jnp.einsum("baz,bz->ba", f_a, z)  # (batch_size, action_size)
        f_a = f_a * self.forward_weight + self.forward_bias
        # q = -jax.nn.log_sigmoid(f_a) - 1.0  # [-1, inf]
        return f_a


class ZeroshotQFunctionBase(QFunction):
    def __init__(
        self,
        puzzle: Puzzle,
        zeroshot_model: nn.Module = ZeroshotQModelBase,
        init_params: bool = True,
        path: str = None,
        **kwargs,
    ):
        self.puzzle = puzzle
        self.is_fixed = puzzle.fixed_target
        self.action_size = puzzle.action_size
        self.model = zeroshot_model(self.action_size, **kwargs)
        self.path = path
        if path is not None:
            if init_params:
                self.params = self.get_new_params()
            else:
                self.params = self.load_model()
        else:
            self.params = self.get_new_params()

    def get_new_params(self):
        dummy_solve_config = self.puzzle.SolveConfig.default()
        dummy_current = self.puzzle.State.default()
        dummy_solve_config_processed = self.pre_process_solve_config(dummy_solve_config)[
            jnp.newaxis, :
        ]
        dummy_current_processed = self.pre_process_state(dummy_current)[jnp.newaxis, :]
        model_params = self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            dummy_solve_config_processed,
            dummy_current_processed,
        )
        return model_params

    def load_model(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            with open(self.path, "rb") as f:
                model_params = pickle.load(f)
            dummy_solve_config = self.puzzle.SolveConfig.default()
            dummy_solve_config_processed = self.pre_process_solve_config(dummy_solve_config)[
                jnp.newaxis, :
            ]
            dummy_current = self.puzzle.State.default()
            dummy_current_processed = self.pre_process_state(dummy_current)[jnp.newaxis, :]
            self.model.apply(
                model_params,
                dummy_current_processed,
                dummy_solve_config_processed,
                training=False,
            )  # check if the params are compatible with the modelWW
        except Exception as e:
            print(f"Error loading model: {e}")
            model_params = self.get_new_params()
        return model_params

    def save_model(self):
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump((self.params, self.goal_params), f)

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

from abc import abstractmethod

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from puzzle.puzzle_base import Puzzle, state_dataclass


# Residual Block
class AutoEncoder(nn.Module):
    latent_size: int = 400

    @nn.compact
    def __call__(self, x0, training=False):
        shape = x0.shape
        flatten = jnp.reshape(x0, shape=(shape[0], -1))
        flatten_size = flatten.shape[1]
        x = nn.Dense(flatten_size)(flatten)
        x = nn.relu(x)
        x = nn.Dense(flatten_size)(x)
        latent = nn.sigmoid(x)
        x = nn.Dense(flatten_size)(latent)
        x = nn.relu(x)
        x = nn.Dense(flatten_size)(x)
        output = jnp.reshape(x, shape)
        return latent, output


class WorldModel(nn.Module):
    latent_size: int
    action_size: int

    @nn.compact
    def __call__(self, latent, training=False):
        x = nn.Dense(500)(latent)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(400 * self.action_size)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.sigmoid(x)
        x = jnp.reshape(x, shape=(x.shape[0], self.action_size, 400))
        return x


class WorldModelPuzzleBase(Puzzle):
    @state_dataclass
    class State:
        """
        The state of the world model puzzle is 'must' be latent.
        It should not be changed in any subclasses.
        """

        latent: jnp.ndarray

    @state_dataclass
    class SolveConfig:
        """
        The solve config of the world model puzzle is 'must' be TargetState.
        It should not be changed in any subclasses.
        """

        TargetState: "WorldModelPuzzleBase.State"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.world_model = WorldModel(latent_size=self.latent_size, action_size=self.action_size)

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        pass

    @abstractmethod
    def get_default_gen(self) -> callable:
        """
        This function should return a callable that takes a state and returns a shape of it.
        function signature: (state: State) -> Dict[str, Any]
        """
        pass

    @abstractmethod
    def get_img_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a image representation of it.
        function signature: (state: State) -> jnp.ndarray
        """
        pass

    @abstractmethod
    def get_solve_config(self, key=None) -> SolveConfig:
        """
        This function should return a solve config.
        """
        pass

    @abstractmethod
    def get_initial_state(self, solve_config: SolveConfig, key=None) -> State:
        """
        This function should return a initial state.
        """
        pass

    def batched_get_neighbours(
        self, solve_config: SolveConfig, states: State, filleds: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        uint8_latent = states.latent
        bit_latent = jax.vmap(self.from_uint8)(uint8_latent)
        next_bit_latent = self.world_model(
            bit_latent, training=False
        )  # (batch_size, action_size, latent_size)
        next_bit_latent = jnp.round(next_bit_latent).astype(jnp.bool_)
        next_bit_latent = jnp.swapaxes(
            next_bit_latent, 0, 1
        )  # (action_size, batch_size, latent_size)
        next_uint8_latent = jax.vmap(jax.vmap(self.to_uint8))(
            next_bit_latent
        )  # (action_size, batch_size, latent_size)
        return (
            self.State(latent=next_uint8_latent),
            jnp.ones((self.action_size, states.latent.shape[0])) * filleds,
        )  # (action_size, batch_size, latent_size), (action_size, batch_size)

    def get_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        states = jax.tree.tree_map(lambda x: x[jnp.newaxis, ...], state)
        filleds = filled[jnp.newaxis, ...]
        return self.batched_get_neighbours(solve_config, states, filleds)

    def batched_is_solved(self, solve_config: SolveConfig, states: State) -> bool:
        """
        This function should return a boolean array that indicates whether the state is the target state.
        """
        return jax.vmap(self.is_solved, in_axes=(None, 0))(solve_config, states)

    def is_solved(self, solve_config: SolveConfig, state: State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        target_state = solve_config.TargetState
        return self.is_equal(state, target_state)

    def to_uint8(self, bit_latent: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        return jnp.packbits(bit_latent, axis=-1, bitorder="little")

    def from_uint8(self, uint8_latent: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        return jnp.unpackbits(uint8_latent, axis=-1, count=self.latent_size, bitorder="little")

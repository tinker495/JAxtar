import pickle

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle, state_dataclass
from puzzle.world_model.util import download_dataset, is_dataset_downloaded


# Residual Block
class AutoEncoder(nn.Module):
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, x0, training=False):
        latent = self.encode(x0, training)
        output = self.decode(latent, training)
        return latent, output

    @nn.compact
    def encode(self, data, training=False):
        shape = data.shape
        data = (data / 255.0) * 2 - 1
        flatten = jnp.reshape(data, shape=(shape[0], -1))
        latent_size = np.prod(self.latent_shape)
        x = nn.Dense(1000)(flatten)
        x = nn.relu(x)
        x = nn.Dense(latent_size)(x)
        x = jnp.reshape(x, shape=(-1, *self.latent_shape))
        latent = nn.sigmoid(x)
        return latent

    @nn.compact
    def decode(self, latent, training=False):
        output_size = np.prod(self.data_shape)
        x = nn.Dense(1000)(latent)
        x = nn.relu(x)
        x = nn.Dense(output_size)(x)
        output = jnp.reshape(x, (-1, *self.data_shape))
        return output


class WorldModel(nn.Module):
    latent_shape: tuple[int, ...]
    action_size: int

    @nn.compact
    def __call__(self, latent, training=False):
        x = (latent - 0.5) * 2.0
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        latent_size = np.prod(self.latent_shape)
        x = nn.Dense(latent_size * self.action_size)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.sigmoid(x)
        x = jnp.reshape(x, shape=(x.shape[0], self.action_size) + self.latent_shape)
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

    def __init__(
        self,
        data_path,
        data_shape,
        latent_shape,
        action_size,
        AE=AutoEncoder,
        WM=WorldModel,
        init_params=True,
        **kwargs,
    ):
        self.data_path = data_path
        self.data_shape = data_shape
        self.latent_shape = latent_shape
        self.action_size = action_size

        class total_model(nn.Module):
            autoencoder: AutoEncoder
            world_model: WorldModel

            @nn.compact
            def __call__(self, x, training=False):
                latent, decoded = self.autoencoder(x, training)
                return self.world_model(latent, training), decoded

            def decode(self, latent, training=False):
                return self.autoencoder.decode(latent, training)

            def encode(self, data, training=False):
                return self.autoencoder.encode(data, training)

            def transition(self, latent, training=False):
                return self.world_model(latent, training)

        self.model = total_model(
            autoencoder=AE(data_shape=self.data_shape, latent_shape=self.latent_shape),
            world_model=WM(latent_shape=self.latent_shape, action_size=self.action_size),
        )

        if init_params:
            self.params = self.get_new_params()

        super().__init__(**kwargs)

    def get_new_params(self):
        dummy_data = jnp.zeros((1, *self.data_shape))
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            dummy_data,
        )

    @classmethod
    def load_model(cls, path: str):

        try:
            with open(path, "rb") as f:
                params = pickle.load(f)
            puzzle = cls(init_params=False)
            dummy_data = jnp.zeros((1, *puzzle.data_shape))
            puzzle.model.apply(
                params,
                dummy_data,
            )  # check if the params are compatible with the model
            puzzle.params = params
        except Exception as e:
            print(f"Error loading model: {e}")
            puzzle = cls()
        return puzzle

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        if not is_dataset_downloaded():
            download_dataset()
        self.inits = jnp.load(self.data_path + "/inits.npy").to_device(jax.devices("gpu")[0])
        self.targets = jnp.load(self.data_path + "/targets.npy").to_device(jax.devices("gpu")[0])
        self.num_puzzles = self.inits.shape[0]

    def get_string_parser(self):
        def parser(
            state: "WorldModelPuzzleBase.State",
            solve_config: "WorldModelPuzzleBase.SolveConfig" = None,
            **kwargs,
        ):
            # Unpack the board before visualization.
            latent = state.latent
            latent = np.array(latent)
            latent = np.reshape(latent, shape=(-1,))
            latent_str = latent.tobytes().hex()
            if len(latent_str) >= 25:
                prefix = latent_str[:10]
                suffix = latent_str[-10:]
                latent_str = prefix + "..." + suffix
            return f"latent: 0x{latent_str}"

        return parser

    def get_default_gen(self) -> callable:
        """
        This function should return a callable that takes a state and returns a shape of it.
        function signature: (state: State) -> Dict[str, Any]
        """

        def default_gen():
            latent_bool = jnp.zeros(self.latent_shape, dtype=jnp.bool_)
            latent_uint8 = self.to_uint8(latent_bool)
            return WorldModelPuzzleBase.State(latent=latent_uint8)

        return default_gen

    def get_img_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a image representation of it.
        function signature: (state: State) -> jnp.ndarray
        """
        import cv2

        def img_parser(state: WorldModelPuzzleBase.State) -> jnp.ndarray:
            latent = jnp.expand_dims(state.latent, axis=0)
            data = self.model.autoencoder.decode(latent).squeeze(0)
            img = cv2.resize(data, (IMG_SIZE, IMG_SIZE))
            return img

        return img_parser

    def get_solve_config(self, key=None) -> SolveConfig:
        """
        This function should return a solve config.
        """
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        data = jnp.expand_dims(self.targets[idx, ...], axis=0)
        latent = self.model.apply(
            self.params, data, training=False, method=self.model.encode
        ).squeeze(0)
        latent = jnp.round(latent).astype(jnp.bool_)
        latent = self.to_uint8(latent)
        return self.SolveConfig(TargetState=self.State(latent=latent))

    def get_initial_state(self, solve_config: SolveConfig, key=None) -> State:
        """
        This function should return a initial state.
        """
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        data = jnp.expand_dims(self.inits[idx, ...], axis=0)
        latent = self.model.apply(
            self.params, data, training=False, method=self.model.encode
        ).squeeze(0)
        latent = jnp.round(latent).astype(jnp.bool_)
        latent = self.to_uint8(latent)
        return self.State(latent=latent)

    def batched_get_neighbours(
        self, solve_config: SolveConfig, states: State, filleds: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        uint8_latent = states.latent
        bit_latent = jax.vmap(self.from_uint8)(uint8_latent)
        next_bit_latent = self.model.apply(
            self.params, bit_latent, training=False, method=self.model.transition
        )  # (batch_size, action_size, latent_size)
        next_bit_latent = jnp.round(next_bit_latent).astype(jnp.bool_)
        next_bit_latent = jnp.swapaxes(
            next_bit_latent, 0, 1
        )  # (action_size, batch_size, latent_size)
        next_uint8_latent = jax.vmap(jax.vmap(self.to_uint8))(
            next_bit_latent
        )  # (action_size, batch_size, latent_size)
        cost = jnp.ones((self.action_size, states.latent.shape[0]), dtype=jnp.float16) * filleds
        return (
            self.State(latent=next_uint8_latent),
            cost,
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
        bit_latent = jnp.reshape(bit_latent, shape=(-1,))
        return jnp.packbits(bit_latent, axis=-1, bitorder="little")

    def from_uint8(self, uint8_latent: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        latent_size = np.prod(self.latent_shape)
        bit_latent = jnp.unpackbits(uint8_latent, axis=-1, count=latent_size, bitorder="little")
        return jnp.reshape(bit_latent, shape=self.latent_shape)

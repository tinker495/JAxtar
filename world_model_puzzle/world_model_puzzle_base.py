import pickle

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from puxle import Puzzle
from puxle.core.puzzle_state import FieldDescriptor, PuzzleState, state_dataclass
from puxle.utils import from_uint8, to_uint8
from puxle.utils.annotate import IMG_SIZE

from helpers.formatting import img_to_colored_str
from neural_util.modules import DTYPE, BatchNorm
from neural_util.util import (
    download_model,
    download_world_model_dataset,
    is_model_downloaded,
    is_world_model_dataset_downloaded,
    round_through_gradient,
)


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, data, training=False):
        shape = data.shape
        data = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        flatten = jnp.reshape(data, shape=(shape[0], -1))
        latent_size = np.prod(self.latent_shape)
        x = nn.Dense(1000, dtype=DTYPE)(flatten)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(latent_size, dtype=DTYPE)(x)
        logits = jnp.reshape(x, shape=(-1, *self.latent_shape))
        return logits


class Decoder(nn.Module):
    data_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, latent, training=False):
        output_size = np.prod(self.data_shape)
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.Dense(1000, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(output_size, dtype=DTYPE)(x)
        output = jnp.reshape(x, (-1, *self.data_shape))
        return output.astype(DTYPE)


# Residual Block
class AutoEncoder(nn.Module):
    data_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]

    def setup(self):
        self.encoder = Encoder(self.latent_shape)
        self.decoder = Decoder(self.data_shape)

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        output = self.decoder(latent, training)
        return latent, output


class WorldModel(nn.Module):
    latent_shape: tuple[int, ...]
    action_size: int

    @nn.compact
    def __call__(self, latent, training=False):
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.Dense(500, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(500, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(500, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        latent_size = np.prod(self.latent_shape)
        logits = nn.Dense(latent_size * self.action_size, dtype=DTYPE)(x)
        logits = jnp.reshape(logits, shape=(x.shape[0], self.action_size) + self.latent_shape)
        return logits


class WorldModelPuzzleBase(Puzzle):

    inits: jnp.ndarray
    targets: jnp.ndarray
    init_state_size: int = 0
    str_parse_img_size: int = 16
    str_parse_img: bool = True

    def define_state_class(self) -> PuzzleState:
        """Defines the state class for WorldModelPuzzleBase using xtructure."""
        str_parser = self.get_string_parser()
        latent_bool = jnp.zeros(self.latent_shape, dtype=jnp.bool_)
        latent_uint8 = to_uint8(latent_bool)
        latent_shape = self.latent_shape

        @state_dataclass
        class State:
            latent: FieldDescriptor[jnp.uint8, latent_uint8.shape, latent_uint8]

            def __str__(self, **kwargs):
                return str_parser(self, **kwargs)

            @property
            def packed(self):
                packed_latent = to_uint8(self.latent)
                return State(latent=packed_latent)

            @property
            def unpacked(self):
                latent = from_uint8(self.latent, latent_shape)
                return State(latent=latent)

        return State

    def __init__(
        self,
        data_path,
        data_shape,
        latent_shape,
        action_size,
        AE=AutoEncoder,
        WM=WorldModel,
        init_params: bool = False,
        path: str = None,
        **kwargs,
    ):
        self.data_path = data_path
        self.data_shape = data_shape
        self.latent_shape = latent_shape
        self.latent_size = int(np.prod(latent_shape))
        if self.latent_size % 8 != 0:
            self.pad_size = int(np.ceil(self.latent_size / 8) * 8 - self.latent_size)
        else:
            self.pad_size = 0
        self.action_size = action_size
        self.path = path

        class total_model(nn.Module):
            autoencoder: AutoEncoder
            world_model: WorldModel

            @nn.compact
            def __call__(self, x, training=False):
                latent, decoded = self.autoencoder(x, training)
                return self.world_model(latent, training), decoded

            def decode(self, latent, training=False):
                return self.autoencoder.decoder(latent, training)

            def encode(self, data, training=False):
                logits = self.autoencoder.encoder(data, training)
                latent = nn.sigmoid(logits)
                return latent

            def transition(self, latent, training=False):
                logits = self.world_model(latent, training)
                latent = nn.sigmoid(logits)
                return latent

            def train_info(self, data, next_data, action, training=True):
                logits = self.autoencoder.encoder(data, training)
                latents = nn.sigmoid(logits)
                rounded_latents = round_through_gradient(latents)
                decoded = self.decode(rounded_latents, training)

                next_logits = self.autoencoder.encoder(next_data, training)
                next_latents = nn.sigmoid(next_logits)
                rounded_next_latents = round_through_gradient(next_latents)
                next_decoded = self.decode(rounded_next_latents, training)

                next_logits_preds = self.world_model(rounded_latents, training)

                action = jnp.reshape(
                    action, (-1,) + (1,) * (next_logits_preds.ndim - 1)
                )  # [batch_size, 1, ...]
                next_logits_pred = jnp.take_along_axis(next_logits_preds, action, axis=1).squeeze(
                    axis=1
                )  # [batch_size, ...]
                next_latents_pred = nn.sigmoid(next_logits_pred)
                rounded_next_latents_pred = round_through_gradient(next_latents_pred)

                return (
                    logits,
                    rounded_latents,
                    decoded,
                    next_logits,
                    rounded_next_latents,
                    next_decoded,
                    next_logits_pred,
                    rounded_next_latents_pred,
                )

        self.model = total_model(
            autoencoder=AE(data_shape=self.data_shape, latent_shape=self.latent_shape),
            world_model=WM(latent_shape=self.latent_shape, action_size=self.action_size),
        )

        if path is not None:
            if init_params:
                self.params = self.get_new_params()
            else:
                self.params = self.load_model()
        else:
            self.params = self.get_new_params()

        super().__init__(**kwargs)

    def get_new_params(self):
        dummy_data = jnp.zeros((1, *self.data_shape))
        return self.model.init(
            jax.random.PRNGKey(np.random.randint(0, 2**32 - 1)),
            dummy_data,
        )

    def load_model(self):
        try:
            if not is_model_downloaded(self.path):
                download_model(self.path)
            with open(self.path, "rb") as f:
                params = pickle.load(f)
            self.model.apply(
                params,
                jnp.zeros((1, *self.data_shape)),
            )  # check if the params are compatible with the model
            return params
        except Exception as e:
            print(f"Error loading model: {e}")
            return self.get_new_params()

    def save_model(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.params, f)

    def data_init(self):
        """
        This function should be called in the __init__ of the subclass.
        If the puzzle need to load dataset, this function should be filled.
        """
        if not is_world_model_dataset_downloaded():
            download_world_model_dataset()
        self.inits = jnp.load(self.data_path + "/inits.npy").to_device(jax.devices("gpu")[0])
        self.targets = jnp.load(self.data_path + "/targets.npy").to_device(jax.devices("gpu")[0])
        self.init_state_size = self.inits.shape[0]

    def get_string_parser(self):
        def parser(
            state: "WorldModelPuzzleBase.State",
            solve_config: "WorldModelPuzzleBase.SolveConfig" = None,
            **kwargs,
        ):
            str = ""
            if self.str_parse_img:
                state_img = state.img(
                    show_target_state_img=False,
                    resize_img=True,
                    target_height=self.str_parse_img_size,
                )
                ascii_img = img_to_colored_str(state_img)
                str += ascii_img + "\n"
            latent = state.latent
            latent = np.array(latent)
            latent = np.reshape(latent, shape=(-1,))
            latent_str = latent.tobytes().hex()
            latent_str_len = len(latent_str)
            if latent_str_len >= 25:
                prefix = latent_str[:10]
                suffix = latent_str[-10:]
                latent_str = prefix + "..." + suffix
            str += f"latent: 0x{latent_str}[{latent_str_len // 2} bytes]"
            return str

        return parser

    def get_img_parser(self) -> callable:
        """
        This function should return a callable that takes a state and returns a image representation of it.
        function signature: (state: State) -> jnp.ndarray
        """
        import cv2

        def img_parser(
            state: WorldModelPuzzleBase.State,
            solve_config: WorldModelPuzzleBase.SolveConfig = None,
            show_target_state_img: bool = True,
            resize_img: bool = True,
            target_height: int = IMG_SIZE[1],
            **kwargs,
        ) -> jnp.ndarray:
            latent = state.unpacked.latent
            latent = jnp.expand_dims(latent, axis=0)
            data = self.model.apply(
                self.params, latent, training=False, method=self.model.decode
            ).squeeze(0)
            data = np.clip(np.array(data * 255.0) / 2.0 + 127.5, 0, 255).astype(np.uint8)
            height, width = data.shape[:2]
            if resize_img:
                width, height = int(target_height * width / height), target_height
                img = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA)
            else:
                img = data
            if solve_config is not None and show_target_state_img:
                latent = solve_config.TargetState.unpacked.latent
                latent = jnp.expand_dims(latent, axis=0)
                data = self.model.apply(
                    self.params, latent, training=False, method=self.model.decode
                ).squeeze(0)
                data = np.clip(np.array(data * 255.0) / 2.0 + 127.5, 0, 255).astype(np.uint8)
                if resize_img:
                    img2 = cv2.resize(
                        data,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    img2 = data
                line = np.ones((10, width, 3), dtype=np.uint8) * 255
                img = np.concatenate([img, line, img2], axis=0)

            return img

        return img_parser

    def get_data(self, key=None) -> tuple[chex.Array, chex.Array]:
        idx = jax.random.randint(key, (), 0, self.init_state_size)
        target_data = jnp.expand_dims(self.targets[idx, ...], axis=0)
        init_data = jnp.expand_dims(self.inits[idx, ...], axis=0)
        return target_data, init_data

    def get_solve_config(self, key=None, data=None) -> Puzzle.SolveConfig:
        """
        This function should return a solve config.
        """
        target_data, _ = data
        latent = self.model.apply(
            self.params, target_data, training=False, method=self.model.encode
        ).squeeze(0)
        latent = jnp.round(latent).astype(jnp.bool_)
        return self.SolveConfig(TargetState=self.State(latent=latent).packed)

    def get_initial_state(
        self, solve_config: Puzzle.SolveConfig, key=None, data=None
    ) -> Puzzle.State:
        """
        This function should return a initial state.
        """
        _, init_data = data
        latent = self.model.apply(
            self.params, init_data, training=False, method=self.model.encode
        ).squeeze(0)
        latent = jnp.round(latent).astype(jnp.bool_)
        return self.State(latent=latent).packed

    def batched_get_neighbours(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[Puzzle.State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        bit_latent = jax.vmap(lambda x: x.unpacked.latent)(states)
        next_bit_latent = self.model.apply(
            self.params, bit_latent, training=False, method=self.model.transition
        )  # (batch_size, action_size, latent_size)
        next_bit_latent = jnp.round(next_bit_latent).astype(jnp.bool_)
        next_bit_latent = jnp.swapaxes(
            next_bit_latent, 0, 1
        )  # (action_size, batch_size, latent_size)
        next_states = self.State(latent=next_bit_latent)
        next_states = jax.vmap(jax.vmap(lambda x: x.packed))(next_states)
        cost = jnp.where(
            filleds,
            jnp.ones((self.action_size, states.latent.shape[0]), dtype=jnp.float16),
            jnp.inf,
        )
        return next_states, cost

    def get_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: Puzzle.State, filled: bool = True
    ) -> tuple[Puzzle.State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        if impossible to move in a direction cost should be inf and State should be same as input state.
        """
        states = state[jnp.newaxis, ...]
        # filleds = filled[jnp.newaxis, ...]
        next_states, costs = self.batched_get_neighbours(solve_config, states, filled)
        return next_states[:, 0], costs[:, 0]

    def batched_is_solved(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        multi_solve_config: bool = False,
    ) -> bool:
        """
        This function should return a boolean array that indicates whether the state is the target state.
        """
        if multi_solve_config:
            return jax.vmap(self.is_solved, in_axes=(0, 0))(solve_configs, states)
        else:
            return jax.vmap(self.is_solved, in_axes=(None, 0))(solve_configs, states)

    def is_solved(self, solve_config: Puzzle.SolveConfig, state: Puzzle.State) -> bool:
        """
        This function should return True if the state is the target state.
        if the puzzle has multiple target states, this function should return
        True if the state is one of the target conditions.
        e.g sokoban puzzle has multiple target states. box's position should
        be the same as the target position but the player's position can be different.
        """
        return state == solve_config.TargetState

    def get_inverse_neighbours(
        self, solve_config: Puzzle.SolveConfig, state: Puzzle.State, filled: bool = True
    ) -> tuple[Puzzle.State, chex.Array]:
        """
        This function should return inverse neighbours and the cost of the move.
        """
        states = state[jnp.newaxis, ...]
        # filleds = filled[jnp.newaxis, ...]
        next_states, costs = self.batched_get_inverse_neighbours(solve_config, states, filled)
        return next_states[:, 0], costs[:, 0]

    def batched_get_inverse_neighbours(
        self,
        solve_configs: Puzzle.SolveConfig,
        states: Puzzle.State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[Puzzle.State, chex.Array]:
        return self.batched_get_neighbours(solve_configs, states, filleds, multi_solve_config)

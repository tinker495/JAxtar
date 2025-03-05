import pickle

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from puzzle.annotate import IMG_SIZE
from puzzle.puzzle_base import Puzzle, state_dataclass
from puzzle.world_model.util import (
    download_dataset,
    download_model,
    img_to_colored_str,
    is_dataset_downloaded,
    is_model_downloaded,
    latents_to_tsne_img,
    latents_to_umap_img,
    round_through_gradient,
)

STR_PARSE_IMG = True
USE_UMAP = False


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, data, training=False):
        shape = data.shape
        data = (data / 255.0) * 2.0 - 1.0
        flatten = jnp.reshape(data, shape=(shape[0], -1))
        latent_size = np.prod(self.latent_shape)
        x = nn.Dense(1000)(flatten)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.swish(x)
        x = nn.Dense(latent_size)(x)
        logits = jnp.reshape(x, shape=(-1, *self.latent_shape))
        return logits


class Decoder(nn.Module):
    data_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, latent, training=False):
        output_size = np.prod(self.data_shape)
        latent = (latent - 0.5) * 2.0
        x = nn.Dense(1000)(latent)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.swish(x)
        x = nn.Dense(output_size)(x)
        output = jnp.reshape(x, (-1, *self.data_shape))
        return output


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


class Projector(nn.Module):
    projection_dim: int = 250

    @nn.compact
    def __call__(self, binary_latent, training=False):
        normalized_latent = (binary_latent - 0.5) * 2.0
        normalized_latent = jnp.reshape(normalized_latent, shape=(normalized_latent.shape[0], -1))
        normalized_latent = jax.lax.stop_gradient(normalized_latent)
        x = nn.Dense(1000)(normalized_latent)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)
        x = nn.Dense(1000)(x)
        x = nn.swish(x)
        x = nn.Dense(
            self.projection_dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(1 / self.projection_dim)),
        )(x)
        return x


class WorldModel(nn.Module):
    latent_shape: tuple[int, ...]
    action_size: int

    @nn.compact
    def __call__(self, latent, training=False):
        x = (latent - 0.5) * 2.0
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.swish(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.swish(x)
        x = nn.Dense(500)(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.swish(x)
        latent_size = np.prod(self.latent_shape)
        x = nn.Dense(latent_size * self.action_size)(x)
        x = jnp.reshape(x, shape=(x.shape[0], self.action_size) + self.latent_shape)
        return x


class WorldModelPuzzleBase(Puzzle):

    inits: jnp.ndarray
    targets: jnp.ndarray
    num_puzzles: int
    set_solveconfig_only_targetstate: bool = True

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
        projected_latent: jnp.ndarray

    def __init__(
        self,
        data_path,
        data_shape,
        latent_shape,
        action_size,
        AE=AutoEncoder,
        WM=WorldModel,
        init_params=True,
        projection_dim=250,
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

        class total_model(nn.Module):
            autoencoder: AutoEncoder
            world_model: WorldModel
            projection_dim: int

            def setup(self):
                self.forward_projector = Projector(self.projection_dim)
                self.backward_projector = Projector(self.projection_dim)

            @nn.compact
            def __call__(self, x, training=False):
                latent, decoded = self.autoencoder(x, training)
                forward_latents = self.world_model(latent, training)
                forward_projected_latent = self.forward_projector(latent, training)
                backward_projected_latent = self.backward_projector(latent, training)
                return (
                    forward_latents,
                    decoded,
                    forward_projected_latent,
                    backward_projected_latent,
                )

            def decode(self, latent, training=False):
                return self.autoencoder.decoder(latent, training)

            def encode(self, data, training=False):
                logits = self.autoencoder.encoder(data, training)
                return nn.sigmoid(logits)

            def transition(self, latent, training=False):
                logits = self.world_model(latent, training)
                return nn.sigmoid(logits)

            def forward_project(self, latent, training=False):
                return self.forward_projector(latent, training)

            def backward_project(self, latent, training=False):
                return self.backward_projector(latent, training)

            def world_model_train_info(self, data, next_data, actions, training=True):

                logits = self.autoencoder.encoder(data, training)
                latent = nn.sigmoid(logits)
                rounded_latent = round_through_gradient(latent)
                decoded = self.autoencoder.decoder(rounded_latent, training)

                next_logits = self.autoencoder.encoder(next_data, training)
                next_latent = nn.sigmoid(next_logits)
                rounded_next_latent = round_through_gradient(next_latent)
                next_decoded = self.autoencoder.decoder(rounded_next_latent, training)

                actions = jnp.reshape(
                    actions, (-1,) + (1,) * (next_latent.ndim)
                )  # [batch_size, 1, ...]

                next_logits_preds = self.world_model(rounded_latent, training)
                next_latent_preds = nn.sigmoid(next_logits_preds)
                rounded_next_latent_preds = round_through_gradient(next_latent_preds)
                next_logits_pred = jnp.take_along_axis(next_logits_preds, actions, axis=1).squeeze(
                    axis=1
                )  # [batch_size, ...]
                rounded_next_latent_pred = jnp.take_along_axis(
                    rounded_next_latent_preds, actions, axis=1
                ).squeeze(
                    axis=1
                )  # [batch_size, ...]

                return (
                    (logits, rounded_latent, decoded),  # for AutoEncoder and WorldModel
                    (
                        next_logits,
                        rounded_next_latent,
                        next_decoded,
                    ),  # for AutoEncoder and WorldModel
                    (
                        next_logits_pred,
                        rounded_next_latent_pred,
                        rounded_next_latent_preds,
                    ),  # for WorldModel
                )

            def get_projections(self, latent, training=False):
                forward_projected_latent = self.forward_project(latent, training)
                backward_projected_latent = self.backward_project(latent, training)
                return forward_projected_latent, backward_projected_latent

        self.model = total_model(
            autoencoder=AE(data_shape=self.data_shape, latent_shape=self.latent_shape),
            world_model=WM(latent_shape=self.latent_shape, action_size=self.action_size),
            projection_dim=projection_dim,
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
            if not is_model_downloaded(path):
                download_model(path)
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

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

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
            if STR_PARSE_IMG:
                state_img = state.img(
                    show_target_state_img=False, resize_img=False, target_height=16
                )
                ascii_img = img_to_colored_str(state_img)
                return ascii_img
            else:
                latent = state.latent
                latent = np.array(latent)
                latent = np.reshape(latent, newshape=(-1,))
                latent_str = latent.tobytes().hex()
                latent_str_len = len(latent_str)
                if latent_str_len >= 25:
                    prefix = latent_str[:10]
                    suffix = latent_str[-10:]
                    latent_str = prefix + "..." + suffix
                return f"latent: 0x{latent_str}[{latent_str_len // 2} bytes]"

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

    def get_solve_config_default_gen(self) -> SolveConfig:
        """
        This function should return a default solve config.
        """
        default_state = self.State.default()
        default_projected_latent = jnp.zeros(self.model.projection_dim, dtype=jnp.float32)

        def default_gen():
            return self.SolveConfig(
                TargetState=default_state, projected_latent=default_projected_latent
            )

        return default_gen

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
            path: list["WorldModelPuzzleBase.State"] = None,
            idx: int = None,
            cost: float = None,
            dist: float = None,
            **kwargs,
        ) -> jnp.ndarray:
            latent = state.latent
            latent = self.from_uint8(latent)
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
                latent = solve_config.TargetState.latent
                latent = self.from_uint8(latent)
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
            if path is not None and idx is not None and len(path) > 1:
                # Visualize the latent space using t-SNE if path is provided
                latents = [self.from_uint8(state.latent) for state in path]
                latents = jnp.stack(latents, axis=0)
                projected_latents = self.model.apply(
                    self.params, latents, training=False, method=self.model.backward_project
                )
                np_projected_latents = np.array(projected_latents)
                if USE_UMAP:
                    visualization_img = latents_to_umap_img(
                        np_projected_latents,
                        idx=idx,
                        img_width=width,
                        title=f"idx: {idx}, cost: {cost:.1f}, dist: {dist:.1f}",
                    )
                else:
                    visualization_img = latents_to_tsne_img(
                        np_projected_latents,
                        idx=idx,
                        img_width=width,
                        title=f"idx: {idx}, cost: {cost:.1f}, dist: {dist:.1f}",
                    )

                # Add the t-SNE visualization to the main image
                if show_target_state_img:
                    # If we're already showing target state, add t-SNE below
                    line = np.ones((10, visualization_img.shape[1], 3), dtype=np.uint8) * 255
                    img = np.concatenate([img, line, visualization_img], axis=0)
                else:
                    # Otherwise, add t-SNE to the right
                    line = np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 255
                    # Ensure tsne_img height matches img height
                    visualization_img_resized = cv2.resize(
                        visualization_img,
                        (visualization_img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                    img = np.concatenate([img, line, visualization_img_resized], axis=1)

            return img

        return img_parser

    def get_data(self, key=None) -> tuple[chex.Array, chex.Array]:
        idx = jax.random.randint(key, (), 0, self.num_puzzles)
        target_data = jnp.expand_dims(self.targets[idx, ...], axis=0)
        init_data = jnp.expand_dims(self.inits[idx, ...], axis=0)
        return target_data, init_data

    def get_solve_config(self, key=None, data=None) -> SolveConfig:
        """
        This function should return a solve config.
        """
        target_data, _ = data
        latent = self.model.apply(
            self.params, target_data, training=False, method=self.model.encode
        )
        projected_latent = self.model.apply(
            self.params, latent, training=False, method=self.model.backward_project
        )[0]
        latent = jnp.round(latent).astype(jnp.bool_)[0]
        latent = self.to_uint8(latent)
        return self.SolveConfig(
            TargetState=self.State(latent=latent), projected_latent=projected_latent
        )

    def get_initial_state(self, solve_config: SolveConfig, key=None, data=None) -> State:
        """
        This function should return a initial state.
        """
        _, init_data = data
        latent = self.model.apply(
            self.params, init_data, training=False, method=self.model.encode
        ).squeeze(0)
        latent = jnp.round(latent).astype(jnp.bool_)
        latent = self.to_uint8(latent)
        return self.State(latent=latent)

    def batched_get_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
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
        cost = jnp.where(
            filleds,
            jnp.ones((self.action_size, states.latent.shape[0]), dtype=jnp.float16),
            jnp.inf,
        )
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
        states = state[jnp.newaxis, ...]
        # filleds = filled[jnp.newaxis, ...]
        next_states, costs = self.batched_get_neighbours(solve_config, states, filled)
        return next_states[:, 0], costs[:, 0]

    def batched_is_solved(
        self, solve_configs: SolveConfig, states: State, multi_solve_config: bool = False
    ) -> bool:
        """
        This function should return a boolean array that indicates whether the state is the target state.
        """
        if multi_solve_config:
            return jax.vmap(self.is_solved, in_axes=(0, 0))(solve_configs, states)
        else:
            return jax.vmap(self.is_solved, in_axes=(None, 0))(solve_configs, states)

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

    def representation_state(self, state: State) -> jnp.ndarray:
        """
        This function should return a representation of the state.
        """
        binary_latent = self.from_uint8(state.latent).astype(jnp.float32)
        binary_latent = jnp.expand_dims(binary_latent, axis=0)
        projected_latent = self.model.apply(
            self.params, binary_latent, training=False, method=self.model.forward_projector
        )
        return projected_latent

    def representation_solve_config(self, solve_config: SolveConfig) -> jnp.ndarray:
        """
        This function should return a representation of the solve config.
        """
        target_latent = self.from_uint8(solve_config.TargetState.latent).astype(jnp.float32)
        target_latent = jnp.expand_dims(target_latent, axis=0)
        projected_latent = self.model.apply(
            self.params, target_latent, training=False, method=self.model.backward_projector
        )
        return projected_latent

    def to_uint8(self, bit_latent: chex.Array) -> chex.Array:
        # from booleans to uint8
        # boolean 32 to uint8 4
        bit_latent = jnp.reshape(bit_latent, shape=(-1,))
        bit_latent = jnp.pad(
            bit_latent, pad_width=(0, self.pad_size), mode="constant", constant_values=0
        )
        return jnp.packbits(bit_latent, axis=-1, bitorder="little")

    def from_uint8(self, uint8_latent: chex.Array) -> chex.Array:
        # from uint8 4 to boolean 32
        bit_latent = jnp.unpackbits(
            uint8_latent, axis=-1, count=self.latent_size, bitorder="little"
        )
        bit_latent = bit_latent[: self.latent_size]
        return jnp.reshape(bit_latent, shape=self.latent_shape)

    def get_inverse_neighbours(
        self, solve_config: SolveConfig, state: State, filled: bool = True
    ) -> tuple[State, chex.Array]:
        """
        This function should return inverse neighbours and the cost of the move.
        """
        states = state[jnp.newaxis, ...]
        # filleds = filled[jnp.newaxis, ...]
        next_states, costs = self.batched_get_inverse_neighbours(solve_config, states, filled)
        return next_states[:, 0], costs[:, 0]

    def batched_get_inverse_neighbours(
        self,
        solve_configs: SolveConfig,
        states: State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[State, chex.Array]:
        return self.batched_get_neighbours(solve_configs, states, filleds, multi_solve_config)

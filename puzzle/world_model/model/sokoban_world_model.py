import chex
import flax.linen as nn
import jax.numpy as jnp

from neural_util.modules import DTYPE, BatchNorm, ConvResBlock
from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, data, training=False):
        x = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        x = nn.Conv(16, (2, 2), strides=(2, 2), dtype=DTYPE)(x)  # (batch_size, 20, 20, 16)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Conv(16, (2, 2), strides=(2, 2), dtype=DTYPE)(x)  # (batch_size, 10, 10, 16)
        x = nn.relu(x)
        logits = nn.Conv(self.latent_shape[-1], (1, 1), strides=(1, 1), dtype=DTYPE)(
            x
        )  # (batch_size, 10, 10, 16)
        return logits


class Decoder(nn.Module):
    data_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, latent, training=False):
        # batch
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2), dtype=DTYPE)(x)  # (batch_size, 20, 20, 16)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2), dtype=DTYPE)(x)  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.Conv(3, (1, 1), strides=(1, 1), dtype=DTYPE)(x)  # (batch_size, 40, 40, 3)
        return x


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
        x = nn.Conv(
            32, (3, 3), strides=(1, 1), kernel_init=nn.initializers.orthogonal(), dtype=DTYPE
        )(
            x
        )  # (batch_size, 10, 10, 32)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Conv(
            32, (3, 3), strides=(1, 1), kernel_init=nn.initializers.orthogonal(), dtype=DTYPE
        )(
            x
        )  # (batch_size, 10, 10, 32)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Conv(
            self.latent_shape[-1] * self.action_size,
            (3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
        )(
            x
        )  # (batch_size, 10, 10, 16 * self.action_size)
        x = jnp.reshape(
            x, shape=(x.shape[0], *self.latent_shape) + (self.action_size,)
        )  # (batch_size, 10, 10, 16, 4)
        logits = jnp.transpose(x, (0, 4, 1, 2, 3))  # (batch_size, 4, 10, 10, 16)
        return logits


class SokobanWorldModel(WorldModelPuzzleBase):

    str_parse_img_size: int = 20

    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/sokoban",
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 16),
            action_size=4,
            AE=AutoEncoder,
            WM=WorldModel,
            **kwargs,
        )

    def batched_get_inverse_neighbours(
        self,
        solve_configs: WorldModelPuzzleBase.SolveConfig,
        states: WorldModelPuzzleBase.State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[WorldModelPuzzleBase.State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        raise NotImplementedError(
            "Sokoban is not reversible,"
            "so sokoban world model's inverse neighbours is not implemented for now\n"
            "Please use '--using_hindsight_target' to train distance"
        )

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        match action:
            case 0:
                return "←"
            case 1:
                return "→"
            case 2:
                return "↑"
            case 3:
                return "↓"
            case _:
                raise ValueError(f"Invalid action: {action}")


class EncoderOptimized(nn.Module):
    latent_shape: tuple[int, int, int]

    @nn.compact
    def __call__(self, data, training=False):
        x = ((data / 255.0) * 2.0 - 1.0).astype(DTYPE)
        x = nn.Conv(
            16, (4, 4), strides=(4, 4), kernel_init=nn.initializers.orthogonal(), dtype=DTYPE
        )(
            x
        )  # (batch_size, 10, 10, 16)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ConvResBlock(16, (1, 1), (1, 1))(x, training)
        logits = nn.Conv(
            self.latent_shape[-1],
            (1, 1),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
        )(
            x
        )  # (batch_size, 10, 10, 2)
        return logits


class DecoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]

    @nn.compact
    def __call__(self, latent, training=False):
        x = ((latent - 0.5) * 2.0).astype(DTYPE)
        x = nn.Conv(16, (1, 1), strides=(1, 1), dtype=DTYPE)(x)  # (batch_size, 10, 10, 16)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = ConvResBlock(16, (1, 1), (1, 1))(x, training)
        x = nn.ConvTranspose(
            16, (4, 4), strides=(4, 4), kernel_init=nn.initializers.orthogonal(), dtype=DTYPE
        )(
            x
        )  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.Conv(
            3, (1, 1), strides=(1, 1), kernel_init=nn.initializers.orthogonal(), dtype=DTYPE
        )(
            x
        )  # (batch_size, 40, 40, 3)
        return x


class AutoEncoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]

    def setup(self):
        self.encoder = EncoderOptimized(self.latent_shape)
        self.decoder = DecoderOptimized(self.data_shape)

    def __call__(self, x0, training=False):
        latent = self.encoder(x0, training)
        output = self.decoder(latent, training)
        return latent, output


class SokobanWorldModelOptimized(WorldModelPuzzleBase):
    """
    This is the optimized version of the sokoban world model.
    sokoban has 4 components so one position could be 2 bits.
    so 10x10x2 is enough for the sokoban world model.
    """

    str_parse_img_size: int = 20

    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/sokoban",
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 2),  # enough for sokoban world model
            action_size=4,
            AE=AutoEncoderOptimized,
            WM=WorldModel,
            **kwargs,
        )

    def batched_get_inverse_neighbours(
        self,
        solve_configs: WorldModelPuzzleBase.SolveConfig,
        states: WorldModelPuzzleBase.State,
        filleds: bool = True,
        multi_solve_config: bool = False,
    ) -> tuple[WorldModelPuzzleBase.State, chex.Array]:
        """
        This function should return a neighbours, and the cost of the move.
        """
        raise NotImplementedError(
            "Sokoban is not reversible,"
            "so sokoban world model's inverse neighbours is not implemented for now\n"
            "Please use '--using_hindsight_target' to train distance"
        )

    def action_to_string(self, action: int) -> str:
        """
        This function should return a string representation of the action.
        """
        match action:
            case 0:
                return "←"
            case 1:
                return "→"
            case 2:
                return "↑"
            case 3:
                return "↓"
            case _:
                raise ValueError(f"Invalid action: {action}")

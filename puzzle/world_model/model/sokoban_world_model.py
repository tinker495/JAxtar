import chex
import flax.linen as nn
import jax.numpy as jnp

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, data, training=False):
        x = (data / 255.0) * 2.0 - 1.0
        x = nn.Conv(16, (2, 2), strides=(2, 2))(x)  # (batch_size, 20, 20, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(16, (2, 2), strides=(2, 2))(x)  # (batch_size, 10, 10, 16)
        x = nn.relu(x)
        x = nn.Conv(self.latent_shape[-1], (1, 1), strides=(1, 1))(x)  # (batch_size, 10, 10, 16)
        latent = nn.sigmoid(x)
        return latent


class Decoder(nn.Module):
    data_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, latent, training=False):
        # batch
        x = (latent - 0.5) * 2.0
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x)  # (batch_size, 20, 20, 16)
        x = nn.relu(x)
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x)  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.Conv(3, (1, 1), strides=(1, 1))(x)  # (batch_size, 40, 40, 3)
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
        x = (latent - 0.5) * 2.0
        x = nn.Conv(32, (3, 3), strides=(1, 1), kernel_init=nn.initializers.orthogonal())(
            x
        )  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), strides=(1, 1), kernel_init=nn.initializers.orthogonal())(
            x
        )  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(
            self.latent_shape[-1] * self.action_size,
            (3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.orthogonal(),
        )(
            x
        )  # (batch_size, 10, 10, 16 * self.action_size)
        x = jnp.reshape(
            x, shape=(x.shape[0], *self.latent_shape) + (self.action_size,)
        )  # (batch_size, 10, 10, 16, 4)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # (batch_size, 4, 10, 10, 16)
        x = nn.sigmoid(x)
        return x


class SokobanWorldModel(WorldModelPuzzleBase):
    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/sokoban",
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 16),
            action_size=4,
            AE=AutoEncoder,
            WM=WorldModel,
            **kwargs
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
            "but sokoban world model's inverse neighbours is not implemented"
        )


# Residual Block
class ResidualBlock(nn.Module):
    channels: int
    kernel_size: tuple[int, int] = (1, 1)
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
            kernel_init=nn.initializers.orthogonal(),
        )(
            x0
        )  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            self.kernel_size,
            strides=self.strides,
            kernel_init=nn.initializers.orthogonal(),
        )(
            x
        )  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = x + x0
        return x


class EncoderOptimized(nn.Module):
    latent_shape: tuple[int, int, int]

    @nn.compact
    def __call__(self, data, training=False):
        x = (data / 255.0) * 2.0 - 1.0
        x = nn.Conv(16, (4, 4), strides=(4, 4), kernel_init=nn.initializers.orthogonal())(
            x
        )  # (batch_size, 10, 10, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = ResidualBlock(16)(x, training)
        x = ResidualBlock(16)(x, training)
        x = nn.Conv(
            self.latent_shape[-1], (1, 1), strides=(1, 1), kernel_init=nn.initializers.orthogonal()
        )(
            x
        )  # (batch_size, 10, 10, 2)
        latent = nn.sigmoid(x)
        return latent


class DecoderOptimized(nn.Module):
    data_shape: tuple[int, int, int]

    @nn.compact
    def __call__(self, latent, training=False):
        x = (latent - 0.5) * 2.0
        x = nn.Conv(16, (1, 1), strides=(1, 1))(x)  # (batch_size, 10, 10, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = ResidualBlock(16)(x, training)
        x = ResidualBlock(16)(x, training)
        x = nn.ConvTranspose(16, (4, 4), strides=(4, 4), kernel_init=nn.initializers.orthogonal())(
            x
        )  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.Conv(3, (1, 1), strides=(1, 1), kernel_init=nn.initializers.orthogonal())(
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

    def __init__(self, **kwargs):

        super().__init__(
            data_path="puzzle/world_model/data/sokoban",
            data_shape=(40, 40, 3),
            latent_shape=(10, 10, 2),  # enough for sokoban world model
            action_size=4,
            AE=AutoEncoderOptimized,
            WM=WorldModel,
            **kwargs
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
            "but sokoban world model's inverse neighbours is not implemented"
        )

import flax.linen as nn
import jax.numpy as jnp

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase

# Residual Block


class ResidualBlock(nn.Module):
    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(16, (1, 1), strides=(1, 1))(x0)  # (batch_size, 10, 10, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(16, (1, 1), strides=(1, 1))(x)  # (batch_size, 10, 10, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = x + x0
        return x


class Encoder(nn.Module):
    latent_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, data, training=False):
        data = (data / 255.0) * 2.0 - 1.0
        x = nn.Conv(16, (4, 4), strides=(4, 4))(data)  # (batch_size, 10, 10, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = ResidualBlock()(x, training)
        x = ResidualBlock()(x, training)
        x = nn.Conv(self.latent_shape[-1], (1, 1), strides=(1, 1))(x)  # (batch_size, 10, 10, 16)
        latent = nn.sigmoid(x)
        return latent


class Decoder(nn.Module):
    data_shape: tuple[int, ...]

    @nn.compact
    def __call__(self, latent, training=False):
        # batch
        x = nn.ConvTranspose(16, (1, 1), strides=(1, 1))(latent)  # (batch_size, 10, 10, 16)
        x = ResidualBlock()(x, training)
        x = ResidualBlock()(x, training)
        x = nn.ConvTranspose(16, (4, 4), strides=(4, 4))(x)  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.ConvTranspose(3, (1, 1), strides=(1, 1))(x)  # (batch_size, 40, 40, 3)
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
        x = nn.Conv(32, (3, 3), strides=(1, 1))(x)  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3), strides=(1, 1))(x)  # (batch_size, 10, 10, 32)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(self.latent_shape[-1] * self.action_size, (3, 3), strides=(1, 1))(
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
            latent_shape=(10, 10, 4),
            action_size=4,
            AE=AutoEncoder,
            WM=WorldModel,
            **kwargs
        )

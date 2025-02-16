import flax.linen as nn
import jax.numpy as jnp

from puzzle.world_model.world_model_puzzle_base import WorldModelPuzzleBase

# Residual Block


class AutoEncoder(nn.Module):
    data_shape: tuple[int, ...] = (40, 40, 3)
    latent_shape: tuple[int, ...] = (10, 10, 4)

    @nn.compact
    def __call__(self, x0, training=False):
        latent = self.encode(x0, training)
        output = self.decode(latent, training)
        return latent, output

    @nn.compact
    def encode(self, data, training=False):
        # (batch_size, 40, 40, 3)
        data = (data / 255.0) * 2 - 1
        x = nn.Conv(16, (2, 2), strides=(2, 2))(data)  # (batch_size, 20, 20, 16)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.Conv(self.latent_shape[-1], (2, 2), strides=(2, 2))(x)  # (batch_size, 10, 10, 16)
        latent = nn.sigmoid(x)
        return latent

    @nn.compact
    def decode(self, latent, training=False):
        # batch
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(latent)  # (batch_size, 20, 20, 16)
        x = nn.relu(x)
        x = nn.ConvTranspose(16, (2, 2), strides=(2, 2))(x)  # (batch_size, 40, 40, 16)
        x = nn.relu(x)
        x = nn.Conv(3, (1, 1))(x)  # (batch_size, 40, 40, 3)
        return x


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
        x = jnp.swapaxes(x, 4, 1)  # (batch_size, 4, 10, 10, 16)
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

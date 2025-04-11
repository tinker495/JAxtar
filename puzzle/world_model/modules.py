import flax.linen as nn
import jax.numpy as jnp

DTYPE = jnp.bfloat16


def BatchNorm(x, training):
    return nn.BatchNorm(momentum=0.9, dtype=DTYPE)(x, use_running_average=not training)


# Residual Block
class ResBlock(nn.Module):
    node_size: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Dense(self.node_size, dtype=DTYPE)(x0)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Dense(self.node_size, dtype=DTYPE)(x)
        x = BatchNorm(x, training)
        return nn.relu(x + x0)


# Conv Residual Block
class ConvResBlock(nn.Module):
    filters: int
    kernel_size: int
    strides: int

    @nn.compact
    def __call__(self, x0, training=False):
        x = nn.Conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="SAME",
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
        )(x0)
        x = BatchNorm(x, training)
        x = nn.relu(x)
        x = nn.Conv(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="SAME",
            kernel_init=nn.initializers.orthogonal(),
            dtype=DTYPE,
        )(x)
        x = BatchNorm(x, training)
        return nn.relu(x + x0)

import jax
from flax import linen as nn

from neural_util.modules import DEFAULT_NORM_FN, DTYPE, ResBlock, swiglu_fn


def vector_augmentation(x, rngkey):
    noise = jax.random.normal(rngkey, x.shape) * 0.1
    return x + noise


class Encoder(nn.Module):
    """Encodes the state into a latent representation."""

    Res_N: int = 2
    latent_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu
    resblock_fn: callable = ResBlock
    use_swiglu: bool = False

    @nn.compact
    def __call__(self, x, training=False):
        if self.use_swiglu:
            x = swiglu_fn(5000, self.activation, self.norm_fn, training)(x)
            x = swiglu_fn(self.latent_dim, self.activation, self.norm_fn, training)(x)
        else:
            x = nn.Dense(5000, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
            x = nn.Dense(self.latent_dim, dtype=DTYPE)(x)
            x = self.norm_fn(x, training)
            x = self.activation(x)
        for _ in range(self.Res_N):
            x = self.resblock_fn(
                self.latent_dim,
                norm_fn=self.norm_fn,
                activation=self.activation,
                use_swiglu=self.use_swiglu,
            )(x, training)
        return x


class ProjectionHead(nn.Module):
    """Projects the latent representation for the contrastive loss."""

    hidden_dim: int = 1000
    output_dim: int = 1000
    activation: str = nn.relu
    use_swiglu: bool = False
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, z, training=False):
        if self.use_swiglu:
            z = swiglu_fn(self.hidden_dim, self.activation, self.norm_fn, training)(z)
            z = swiglu_fn(self.hidden_dim, self.activation, self.norm_fn, training)(z)
        else:
            z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
            z = self.activation(z)
            z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
            z = self.activation(z)
        z = nn.Dense(self.output_dim, dtype=DTYPE)(z)
        return z


class TransitionModel(nn.Module):
    """Predicts the next latent representations for all possible actions."""

    action_size: int
    latent_dim: int = 1000
    hidden_dim: int = 1000
    activation: str = nn.relu
    use_swiglu: bool = False
    norm_fn: callable = DEFAULT_NORM_FN

    @nn.compact
    def __call__(self, z, training=False):
        if self.use_swiglu:
            z = swiglu_fn(self.hidden_dim, self.activation, self.norm_fn, training)(z)
            z = swiglu_fn(self.hidden_dim, self.activation, self.norm_fn, training)(z)
        else:
            z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
            z = self.activation(z)
            z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
            z = self.activation(z)
        z = nn.Dense(self.latent_dim * self.action_size, dtype=DTYPE)(z)
        z = z.reshape((-1, self.action_size, self.latent_dim))
        return z

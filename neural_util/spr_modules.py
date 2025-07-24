from flax import linen as nn

from neural_util.modules import DEFAULT_NORM_FN, DTYPE, ResBlock


class Encoder(nn.Module):
    """Encodes the state into a latent representation."""

    Res_N: int = 2
    latent_dim: int = 1000
    norm_fn: callable = DEFAULT_NORM_FN
    activation: str = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(5000, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        x = self.activation(x)
        x = nn.Dense(self.latent_dim, dtype=DTYPE)(x)
        x = self.norm_fn(x, training)
        x = self.activation(x)
        for _ in range(self.Res_N):
            x = ResBlock(self.latent_dim, norm_fn=self.norm_fn, activation=self.activation)(
                x, training
            )
        return x


class ProjectionHead(nn.Module):
    """Projects the latent representation for the contrastive loss."""

    hidden_dim: int = 1000
    output_dim: int = 1000
    activation: str = nn.relu

    @nn.compact
    def __call__(self, z):
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

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
        z = self.activation(z)
        z = nn.Dense(self.hidden_dim, dtype=DTYPE)(z)
        z = self.activation(z)
        z = nn.Dense(self.latent_dim * self.action_size, dtype=DTYPE)(z)
        z = z.reshape((-1, self.action_size, self.latent_dim))
        return z

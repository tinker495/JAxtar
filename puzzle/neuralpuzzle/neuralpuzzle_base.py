import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from puzzle.puzzle_base import Puzzle, state_dataclass


class VQVAE(nn.Module):
    size: int  # Expected image size (height and width)
    latent_dim: int = 32
    num_embeddings: int = 256

    @nn.compact
    def __call__(self, x, training=False):
        # Ensure the input image has the expected dimensions: (batch, size, size, channels)
        chex.assert_equal(x.shape[1], self.size)
        chex.assert_equal(x.shape[2], self.size)
        in_channels = x.shape[-1]

        # Encoder: Downsample the image and extract features.
        h = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        h = nn.relu(h)
        h = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(h)
        h = nn.relu(h)
        h = nn.Conv(features=self.latent_dim, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(h)

        # Vector Quantization: Quantize the latent representations.
        h_shape = h.shape  # (batch, height, width, latent_dim)
        h_flat = jnp.reshape(h, (-1, self.latent_dim))
        codebook = self.param(
            "codebook",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            (self.num_embeddings, self.latent_dim),
        )
        distances = (
            jnp.sum(h_flat**2, axis=1, keepdims=True)
            - 2 * jnp.dot(h_flat, codebook.T)
            + jnp.sum(codebook**2, axis=1)
        )
        encoding_indices = jnp.argmin(distances, axis=1)  # shape: (batch, )
        _h_q = jnp.take(codebook, encoding_indices, axis=0)
        _h_q = jnp.reshape(_h_q, h_shape)

        # Straight-through estimator for gradient propagation.
        h_q = h + jax.lax.stop_gradient(_h_q - h)

        # Decoder: Reconstruct the image from the quantized latent representation.
        d = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(h_q)
        d = nn.relu(d)
        d = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(d)
        d = nn.relu(d)
        d = nn.ConvTranspose(
            features=in_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
        )(d)
        return _h_q, d

    def loss(self, x, training=True):
        """
        Compute the reconstruction loss for the VQVAE.
        This loss is defined as the mean squared error between the input and its reconstruction.
        """
        reconstructed = self(x, training=training)
        loss_value = jnp.mean((x - reconstructed) ** 2)
        return loss_value


class NeuralPuzzleBase(Puzzle):
    """
    # NeuralPuzzleBase leverages neural network techniques, such as variational autoencoders (VAE),
    # to transform the puzzle environment into a latent representation. This latent encoding
    # fully simulates the world model of the puzzle, enabling a comprehensive heuristic evaluation.
    # It assumes that the Puzzle.State class provides a default() method and a size attribute to
    # facilitate this transformation.
    """

    size: int

    @state_dataclass
    class State:
        codebook: chex.Array
        # codebook is a flattened array of the puzzle state.
        # The size of the codebook array is determined by the puzzle's size.

    def __init__(self, model: nn.Module = VQVAE(), init_params: bool = True):
        self.model = model
        if init_params:
            self.params = self.get_new_params()

    def get_new_params(self):
        # Initialize the neural network parameters using a dummy state.
        # The dummy state is assumed to have a flat representation of size self.State.size.
        return self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, self.State.size)))

    def predict(self, state, training=False):
        # Convert state to a JAX array (if necessary) and reshape to (1, state_size).
        input_state = jnp.array(state).reshape((1, self.State.size))
        return self.model.apply(self.params, input_state, training=training)

    def get_neighbors(self, state):
        # Generate neighbor states by applying transitions to the discretized latent vector.
        # Assumes state has a 'latent' attribute and
        # a 'from_latent' class method to create a new state from a latent vector.
        latent = state.latent
        codebook_size = getattr(
            self.State, "codebook_size", 10
        )  # Default codebook size if not provided by State
        neighbors = []
        for i, val in enumerate(latent):
            # Transition: decrement if possible
            if val > 0:
                new_latent = latent.copy() if hasattr(latent, "copy") else list(latent)
                new_latent[i] = val - 1
                neighbors.append(self.State.from_latent(new_latent))
            # Transition: increment if possible
            if val < codebook_size - 1:
                new_latent = latent.copy() if hasattr(latent, "copy") else list(latent)
                new_latent[i] = val + 1
                neighbors.append(self.State.from_latent(new_latent))
        return neighbors

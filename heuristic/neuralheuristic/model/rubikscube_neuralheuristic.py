import chex
import jax.numpy as jnp
from flax import linen as nn
from puxle import RubiksCube

from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase
from neural_util.modules import DTYPE, HEAD_DTYPE


def _remove_face_centers(flatten_face: chex.Array, n: int) -> chex.Array:
    """Drop centre stickers from the flattened cube faces."""
    face_area = n * n
    total_len = flatten_face.shape[0]
    if face_area == 0:
        raise ValueError("Cube dimension must be positive.")
    if total_len % face_area != 0:
        raise ValueError("Flattened face length is incompatible with cube size.")

    if n % 2 == 0:
        return flatten_face

    num_faces = total_len // face_area
    centre_index = (n // 2) * n + (n // 2)
    indices_before = jnp.arange(centre_index, dtype=jnp.int32)
    indices_after = jnp.arange(centre_index + 1, face_area, dtype=jnp.int32)
    gather_indices = jnp.concatenate([indices_before, indices_after], axis=0)

    faces = flatten_face.reshape((num_faces, face_area))
    faces_without_centre = jnp.take(faces, gather_indices, axis=1)
    return faces_without_centre.reshape(num_faces * (face_area - 1))


def _identity_norm(x: chex.Array, training: bool = False) -> chex.Array:
    del training
    return x


def _unused_resblock(*args, **kwargs):
    del args, kwargs
    return None


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
        residual = x
        attn_input = nn.LayerNorm(dtype=jnp.float32)(residual).astype(DTYPE)
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            dtype=DTYPE,
            deterministic=not training,
        )(attn_input)
        if self.dropout_rate > 0.0:
            attn_output = nn.Dropout(rate=self.dropout_rate)(attn_output, deterministic=not training)
        x = residual + attn_output

        residual = x
        mlp_input = nn.LayerNorm(dtype=jnp.float32)(residual).astype(DTYPE)
        mlp_output = nn.Dense(self.mlp_dim, dtype=DTYPE)(mlp_input)
        mlp_output = nn.gelu(mlp_output)
        if self.dropout_rate > 0.0:
            mlp_output = nn.Dropout(rate=self.dropout_rate)(mlp_output, deterministic=not training)
        mlp_output = nn.Dense(self.embed_dim, dtype=DTYPE)(mlp_output)
        if self.dropout_rate > 0.0:
            mlp_output = nn.Dropout(rate=self.dropout_rate)(mlp_output, deterministic=not training)
        return residual + mlp_output


class RubiksCubeTransformerModel(nn.Module):
    num_classes: int
    seq_len: int
    embed_dim: int = 128
    num_heads: int = 2
    num_layers: int = 2
    mlp_dim: int = 512
    dropout_rate: float = 0.0
    # Compatibility placeholders for NeuralHeuristicBase kwargs.
    norm_fn: callable = _identity_norm
    activation: callable = nn.relu
    resblock_fn: callable = _unused_resblock
    use_swiglu: bool = False

    @nn.compact
    def __call__(self, tokens: chex.Array, training: bool = False) -> chex.Array:
        tokens = tokens.astype(jnp.int32)
        embeddings = nn.Embed(
            num_embeddings=self.num_classes,
            features=self.embed_dim,
            dtype=DTYPE,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(tokens)
        batch_size = embeddings.shape[0]

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim),
        )
        cls_tokens = jnp.tile(cls_token, (batch_size, 1, 1)).astype(DTYPE)
        x = jnp.concatenate([cls_tokens, embeddings], axis=1)

        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.seq_len + 1, self.embed_dim),
        )
        x = x + pos_embedding[jnp.newaxis, :, :].astype(DTYPE)

        for layer_idx in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{layer_idx}",
            )(x, training=training)

        x = nn.LayerNorm(dtype=jnp.float32)(x)
        cls_representation = x[:, 0].astype(DTYPE)
        head = nn.Dense(self.embed_dim, dtype=DTYPE)(cls_representation)
        head = nn.relu(head)
        head = nn.Dense(1, dtype=HEAD_DTYPE)(head)
        return head


class _BaseRubiksCubeTransformerHeuristic(NeuralHeuristicBase):
    def __init__(
        self,
        puzzle: RubiksCube,
        *,
        include_target: bool,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        mlp_dim: int = 512,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        self._use_color_embedding = getattr(puzzle, "color_embedding", True)
        tile_count = puzzle.size * puzzle.size
        self._num_tile_classes = 6 if self._use_color_embedding else 6 * tile_count
        total_stickers = 6 * tile_count
        base_indices = jnp.arange(total_stickers, dtype=jnp.int32)
        self._tokens_per_state = _remove_face_centers(base_indices, puzzle.size).shape[0]
        self._include_target = include_target
        seq_len = self._tokens_per_state * (2 if include_target else 1)
        super().__init__(
            puzzle,
            model=RubiksCubeTransformerModel,
            num_classes=self._num_tile_classes,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def pre_process(
        self, solve_config: RubiksCube.SolveConfig, current: RubiksCube.State
    ) -> chex.Array:
        current_faces = current.unpacked.faces.flatten()
        current_tokens = _remove_face_centers(current_faces, self.puzzle.size).astype(jnp.int32)
        if not self._include_target:
            return current_tokens

        target_faces = solve_config.TargetState.unpacked.faces.flatten()
        target_tokens = _remove_face_centers(target_faces, self.puzzle.size).astype(jnp.int32)
        return jnp.concatenate([target_tokens, current_tokens], axis=0)


class RubiksCubeNeuralHeuristic(_BaseRubiksCubeTransformerHeuristic):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, include_target=False, **kwargs)


class RubiksCubeRandomNeuralHeuristic(_BaseRubiksCubeTransformerHeuristic):
    def __init__(self, puzzle: RubiksCube, **kwargs):
        super().__init__(puzzle, include_target=True, **kwargs)

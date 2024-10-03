from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx


class _MlpBlock(nnx.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        dropout_rate: float,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dense1 = nnx.Linear(
            in_features=in_features,
            out_features=mlp_dim,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        self.dense2 = nnx.Linear(
            in_features=mlp_dim,
            out_features=out_features,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, *, deterministic: bool):
        """Applies Transformer MlpBlock module."""
        # First layer
        x = self.dropout1(nnx.gelu(self.dense1(x)), deterministic=deterministic)
        # Second layer
        x = self.dropout2(self.dense2(x), deterministic=deterministic)
        return x


class _AttentionBlock(nnx.Module):
    """Attention block.

    Attributes
    ----------
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.firstlayernorm = nnx.LayerNorm(num_features=in_features, rngs=rngs)
        self.secondlayernorm = nnx.LayerNorm(num_features=in_features, rngs=rngs)
        self.multihead = nnx.MultiHeadAttention(
            in_features=in_features,
            kernel_init=nnx.initializers.xavier_uniform(),
            num_heads=num_heads,
            broadcast_dropout=False,
            dropout_rate=attention_dropout_rate,
            decode=False,
            rngs=rngs,
        )
        self.mlp = _MlpBlock(
            in_features=in_features,
            out_features=out_features,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, inputs, *, deterministic: bool):
        """Applies Encoder1DBlock module.

        Args:
        ----
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
        -------
          output after transformer encoder block.

        """
        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = inputs
        x = self.firstlayernorm(x)
        x = self.multihead(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = self.secondlayernorm(x)
        y = self.mlp(y, deterministic=deterministic)

        return x + y


@dataclass
class Architecture:
    num_classes: int
    channels: int
    patches_size: int
    num_patches: int
    embed_dim: int
    layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float


class ViT(nnx.Module):
    """VisionTransformer."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(
                rngs.params(),
                shape=(1, 1 + architecture.num_patches, architecture.embed_dim),
            )
        )
        self.cls_token = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(
                rngs.params(), shape=(1, 1, architecture.embed_dim)
            )
        )
        self.patch_embedding = nnx.Conv(
            in_features=architecture.channels,
            out_features=architecture.embed_dim,
            kernel_size=(architecture.patches_size, architecture.patches_size),
            strides=architecture.patches_size,
            padding="VALID",
            rngs=rngs,
        )
        self.transformer_layers = []
        for _ in range(architecture.layers):
            self.transformer_layers.append(
                _AttentionBlock(
                    in_features=architecture.embed_dim,
                    out_features=architecture.embed_dim,
                    mlp_dim=architecture.mlp_dim,
                    num_heads=architecture.num_heads,
                    dropout_rate=architecture.dropout_rate,
                    attention_dropout_rate=architecture.attention_dropout_rate,
                    rngs=rngs,
                )
            )
        self.final_layer = nnx.Linear(
            in_features=architecture.embed_dim,
            out_features=architecture.num_classes,
            rngs=rngs,
        )

    def __call__(self, inputs, *, train: bool):
        x = self.patch_embedding(inputs)
        n, h, w, c = x.shape

        x = jnp.reshape(x, [n, h * w, c])

        # Add CLS token and positional encoding
        cls_token = jnp.repeat(self.cls_token.value, n, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embedding[:, : h * w + 1]

        for attention_block in self.transformer_layers:
            x = attention_block(x, deterministic=not train)

        x = x[:, 0]

        x = self.final_layer(x)

        return x

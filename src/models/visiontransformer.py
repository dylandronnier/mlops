from typing import Optional

import jax.numpy as jnp
from flax import nnx


class MlpBlock(nnx.Module):
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

    def __call__(self, x, *, deterministic):
        """Applies Transformer MlpBlock module."""
        # First layer
        x = self.dropout1(nnx.gelu(self.dense1(x)), deterministic=deterministic)
        # Second layer
        x = self.dropout2(self.dense2(x), deterministic=deterministic)
        return x


class Encoder1DBlock(nnx.Module):
    """Transformer encoder layer.

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
            rngs=rngs,
        )
        self.mlp = MlpBlock(
            in_features=in_features,
            out_features=out_features,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, inputs, *, deterministic):
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


class PosParams(nnx.Param):
    pass


class VisionTransformer(nnx.Module):
    """VisionTransformer."""

    def __init__(
        self,
        num_classes: int,
        patches_size: int,
        layers: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float,
        attendion_dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.pos_embedding = PosParams(jnp.zeros((patches_size, patches_size)))
        self.patch_embedding = nnx.Conv(
            in_features=1,
            out_features=patches_size,
            kernel_size=patches_size,
            strides=patches_size,
            padding="VALID",
            rngs=rngs,
        )
        self.transformer_layers = []
        for _ in range(layers):
            self.transformer_layers.append(
                Encoder1DBlock(
                    in_features=patches_size,
                    out_features=patches_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attendion_dropout_rate,
                    rngs=rngs,
                )
            )
        self.final_layer = nnx.Linear(
            in_features=patches_size, out_features=num_classes, rngs=rngs
        )

    def __call__(self, inputs, *, train: bool):

        x = self.patch_embedding(inputs)
        n, h, w, c = x.shape

        x = jnp.reshape(x, [n, h * w, c])

        for t in self.transformer_layers:
            x = t(x, deterministic=not train)

        x = x[:, 0]

        x = self.final_layer(x)

        return x

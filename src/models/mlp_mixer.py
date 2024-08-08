import jax.numpy as jnp
from flax import nnx
from jax import Array


class MlpBlock(nnx.Module):
    """Multi-layer perceptron block."""

    def __init__(self, features: int, mlp_dim: int, *, rngs: nnx.Rngs) -> None:
        self.first_dense_layer = nnx.Linear(
            in_features=features, out_features=mlp_dim, rngs=rngs
        )

        self.second_dense_layer = nnx.Linear(
            in_features=mlp_dim, out_features=features, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        # Dense layer with gelu activation function
        x = nnx.gelu(self.first_dense_layer(x))

        # Apply the second dense layer
        return self.second_dense_layer(x)


class MixerBlock(nnx.Module):
    """Basic Mlp-Mixer block."""

    def __init__(
        self,
        channel_dim: int,
        token_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.first_layer_norm = nnx.LayerNorm(num_features=channel_dim, rngs=rngs)
        self.second_layer_norm = nnx.LayerNorm(num_features=channel_dim, rngs=rngs)
        self.token_mixing = MlpBlock(
            features=token_dim, mlp_dim=token_mlp_dim, rngs=rngs
        )
        self.channel_mixing = MlpBlock(
            features=channel_dim, mlp_dim=channel_mlp_dim, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        y = self.first_layer_norm(x)
        y = jnp.swapaxes(y, 1, 2)
        y = self.token_mixing(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = self.second_layer_norm(x)
        return x + self.channel_mixing(y)


class MlpMixer(nnx.Module):
    """Mlp Mixer network from Tolstikhin et al."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_blocks: int,
        patches_size: int,
        embed_dim: int,
        image_size: int,
        token_mlp_dim: int,
        channels_mlp_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.patch_embedding = nnx.Conv(
            in_features=channels,
            out_features=embed_dim,
            kernel_size=(patches_size, patches_size),
            strides=(patches_size, patches_size),
            padding="VALID",
            rngs=rngs,
        )
        self.mixer_blocks = []
        for _ in range(num_blocks):
            self.mixer_blocks.append(
                MixerBlock(
                    channel_dim=embed_dim,
                    token_dim=(image_size // patches_size) ** 2,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channels_mlp_dim,
                    rngs=rngs,
                )
            )

        self.pre_head_layer_norm = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.head = nnx.Linear(
            in_features=embed_dim, out_features=num_classes, rngs=rngs
        )

    def __call__(self, x, train: bool) -> Array:
        # Patches projection as tokens
        x = self.patch_embedding(x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # MLP mixer layers
        for block in self.mixer_blocks:
            x = block(x)

        # Apply pre-head layer
        x = self.pre_head_layer_norm(x)
        x = jnp.mean(x, axis=1)

        # Final head layer
        return self.head(x)

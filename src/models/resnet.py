from dataclasses import dataclass

from flax import nnx
from jax import Array
from jax.numpy import mean

from models._basic_cnn_block import BasicBlock


class _ResNetBlock(nnx.Module):
    """Residual Block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strides: tuple[int, int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Construct a Residual Block.

        Args:
        ----
            in_features (int): Number of input channels.
            out_features (int): Number of output channels.
            rngs (nnx.Rngs): Key for the random initialization of the paramters.

        """
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.bn1 = nnx.BatchNorm(
            epsilon=1e-5, momentum=0.9, num_features=out_features, rngs=rngs
        )

        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.bn2 = nnx.BatchNorm(
            epsilon=1e-5,
            momentum=0.9,
            num_features=out_features,
            rngs=rngs,
            scale_init=nnx.initializers.zeros_init(),
        )

        if in_features != out_features or strides != (1, 1):
            self.proj = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                rngs=rngs,
            )
            self.proj_norm = nnx.BatchNorm(
                epsilon=1e-5,
                momentum=0.9,
                num_features=out_features,
                rngs=rngs,
            )

    def __call__(self, x: Array):
        """Run Residual Block.

        Args:
        ----
            x (tensor): Input tensor of shape [N, H, W, C].

        Returns:
        -------
            (tensor): Output shape of shape [N, H', W', features].

        """
        out = nnx.relu(self.bn1(self.conv1(x)))

        out = nnx.relu(self.bn2(self.conv2(out)))

        if x.shape != out.shape:
            x = self.proj_norm(self.proj(x))

        return nnx.relu(out + x)


@dataclass
class Architecture:
    stage_sizes: list[int]
    num_filers: int = 64


class NeuralNetwork(nnx.Module):
    """Residual Neural Network."""

    def __init__(
        self, arch: Architecture, *, channels: int, num_classes: int, rngs: nnx.Rngs
    ) -> None:
        # Basic block for first layer
        self._basic = BasicBlock(
            in_features=channels,
            out_features=arch.num_filers,
            kernel_size=(3, 3),
            rngs=rngs,
        )

        # Residual blocks
        self._resnetblocks = []
        for i, block_size in enumerate(arch.stage_sizes):
            strides = (2, 2) if i > 0 else (1, 1)
            self._resnetblocks.append(
                _ResNetBlock(
                    in_features=arch.num_filers * 2 ** max(i - 1, 0),
                    out_features=arch.num_filers * 2**i,
                    strides=strides,
                    rngs=rngs,
                )
            )
            for _ in range(1, block_size):
                self._resnetblocks.append(
                    _ResNetBlock(
                        in_features=arch.num_filers * 2**i,
                        out_features=arch.num_filers * 2**i,
                        strides=(1, 1),
                        rngs=rngs,
                    )
                )

        # Fully connected last layer
        self._fullyconnected = nnx.Linear(
            in_features=arch.num_filers * 2 ** (len(arch.stage_sizes) - 1),
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x):
        # Apply first CNN layer followed by MaxPooling
        x = self._basic(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Apply the residual blocks
        for b in self._resnetblocks:
            x = b(x)

        # Mean
        x = mean(x, axis=(1, 2))

        # Apply the fully connected layer and return the result
        return self._fullyconnected(x)

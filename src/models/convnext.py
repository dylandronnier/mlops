from dataclasses import dataclass

from flax import nnx
from jax import Array
from jax.numpy import mean

from models._basic_cnn_block import BasicBlock


class _ConvNeXTBlock(nnx.Module):
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
            kernel_size=(7, 7),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(epsilon=1e-5, num_features=out_features, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            strides=strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            strides=strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
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

    def __call__(self, x: Array, train: bool = True):
        """Run Residual Block.

        Args:
        ----
            x (tensor): Input tensor of shape [N, H, W, C].
            train (bool): Training mode.

        Returns:
        -------
            (tensor): Output shape of shape [N, H', W', features].

        """
        out = self.conv3(nnx.gelu(self.conv2(self.ln(self.conv1(x)))))

        # out = nnx.relu(self.bn2(self.conv2(out), use_running_average=train))

        if x.shape != out.shape:
            x = self.proj_norm(self.proj(x))

        return nnx.relu(out + x)


@dataclass
class Architecture:
    num_classes: int
    channels: int
    stage_sizes: list[int]
    num_filers: int = 64


class NeuralNetwork(nnx.Module):
    """Residual Neural Network."""

    def __init__(self, arch: Architecture, *, rngs: nnx.Rngs) -> None:
        # Basic block for first layer
        self._basic = BasicBlock(
            in_features=arch.channels,
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
            out_features=arch.num_classes,
            rngs=rngs,
        )

    def __call__(self, x, train: bool = False):
        # Apply first CNN layer followed by MaxPooling
        x = self._basic(x, train)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Apply the residual blocks
        for b in self._resnetblocks:
            x = b(x, train)

        # Mean
        x = mean(x, axis=(1, 2))

        # Apply the fully connected layer and return the result
        return self._fullyconnected(x)

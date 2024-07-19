from flax import nnx
from jax import Array
from jax.numpy import mean

from models._basic_cnn_block import BasicBlock3


class ResBlock(nnx.Module):

    """Residual Block."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Construct a Residual Block

        Args:
        ----
            in_features (int): Number of input channels.
            out_features (int): Number of output channels.
            rngs: Key for the random initialization of the paramters.

        """
        self.first_conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.first_batch_norm = nnx.BatchNorm(
            epsilon=1e-5, momentum=0.9, num_features=out_features, rngs=rngs
        )

        self.second_conv = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.second_batch_norm = nnx.BatchNorm(
            epsilon=1e-5, momentum=0.9, num_features=out_features, rngs=rngs
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
        residual = x

        x = nnx.relu(
            self.first_batch_norm(self.first_conv(x), use_running_average=train)
        )

        x = nnx.relu(
            self.second_batch_norm(self.second_conv(x), use_running_average=train)
        )

        x += residual

        return x


class Resnet9(nnx.Module):

    """Resnet9 is a Residual Neural Network composed of ."""

    def __init__(self, num_classes: int, *, rngs: nnx.Rngs) -> None:
        self.first_basic = BasicBlock3(in_features=3, out_features=64, rngs=rngs)
        self.second_basic = BasicBlock3(in_features=64, out_features=128, rngs=rngs)

        self.first_res = ResBlock(
            in_features=128,
            out_features=128,
            rngs=rngs,
        )
        self.third_basic = BasicBlock3(in_features=128, out_features=256, rngs=rngs)

        self.fourth_basic = BasicBlock3(in_features=256, out_features=256, rngs=rngs)
        self.second_res = ResBlock(in_features=256, out_features=256, rngs=rngs)
        self.last_layer = nnx.Linear(
            in_features=256, out_features=num_classes, rngs=rngs
        )

    def __call__(self, x, train: bool = False):
        x = self.second_basic(self.first_basic(x, train), train)
        x = self.first_res(x, train)
        x = nnx.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        x = self.fourth_basic(self.third_basic(x, train), train)
        x = self.second_res(x, train)
        x = nnx.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        x = mean(x, axis=(1, 2))
        return self.last_layer(x)

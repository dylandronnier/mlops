from flax import nnx


class BasicBlock3(nnx.Module):

    """Basic CNN block with batch norm."""

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
        # Convolution layer with "same" padding, kernel size=3 and strides=1.
        self.convolution_layer = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        self.batch_norm = nnx.BatchNorm(
            momentum=0.9, num_features=out_features, rngs=rngs
        )

    def __call__(self, x, train: bool = True):
        """Run Residual Block.

        Args:
        ----
            x (tensor): Input tensor of shape [N, H, W, C].
            train (bool): Training mode.

        Returns:
        -------
            (tensor): Output shape of shape [N, H', W', features].

        """
        return nnx.relu(self.batch_norm(self.convolution_layer(x), use_running_average=train))


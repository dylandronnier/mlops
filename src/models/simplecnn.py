from dataclasses import dataclass

from flax import nnx
from flax.nnx.nnx.nn.linear import Linear
from jax import Array

from models._basic_cnn_block import BasicBlock


@dataclass
class Architecture:
    cnn_fliters: list[int]
    layers_sizes: list[int]
    fc_layers_widths: list[int]
    intermediate_size: int


class _SimpleCNNBlock(nnx.Module):
    def __init__(
        self, in_features: int, out_features: int, nb_conv_layers: int, rngs: nnx.Rngs
    ) -> None:
        assert nb_conv_layers > 0
        self.cnn_layers = []
        self.cnn_layers.append(
            BasicBlock(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(3, 3),
                rngs=rngs,
            )
        )
        for _ in range(nb_conv_layers):
            self.cnn_layers.append(
                BasicBlock(
                    in_features=out_features,
                    out_features=out_features,
                    kernel_size=(3, 3),
                    rngs=rngs,
                )
            )

    def __call__(self, x: Array) -> Array:
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        x = nnx.max_pool(inputs=x, window_shape=(2, 2), strides=(2, 2))
        return x


class SimpleCNN(nnx.Module):
    """A simple CNN model."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        channels: int,
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        i = channels
        self._cnn_part = list()
        for f, nb in zip(architecture.cnn_fliters, architecture.layers_sizes):
            self._cnn_part.append(
                _SimpleCNNBlock(
                    in_features=i, out_features=f, nb_conv_layers=nb, rngs=rngs
                )
            )
            i = f

        # self.second = _SimpleCNNBlock(
        #     in_features=32, out_features=64, nb_conv_layers=2, rngs=rngs
        # )

        # self.third = _SimpleCNNBlock(
        #     in_features=64, out_features=128, nb_conv_layers=2, rngs=rngs
        # )

        # self.fourth = _SimpleCNNBlock(
        #     in_features=128, out_features=256, nb_conv_layers=3, rngs=rngs
        # )

        # self.fifth = _SimpleCNNBlock(
        #     in_features=256, out_features=512, nb_conv_layers=3, rngs=rngs
        # )

        self._fully_connected = list()
        i = architecture.intermediate_size
        for nb in architecture.fc_layers_widths:
            self._fully_connected.append(
                Linear(in_features=i, out_features=nb, rngs=rngs)
            )
            i = nb
        self._head = nnx.Linear(in_features=i, out_features=num_classes, rngs=rngs)

        # self.linear1 = nnx.Linear(in_features=512, out_features=512, rngs=rngs)
        # self.do1 = nnx.Dropout(0.5, deterministic=True, rngs=rngs)
        # self.linear2 = nnx.Linear(in_features=512, out_features=256, rngs=rngs)
        # self.do2 = nnx.Dropout(0.5, deterministic=True, rngs=rngs)
        # self.head = nnx.Linear(in_features=256, out_features=num_classes, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        for l in self._cnn_part:
            x = l(x)

        x = x.reshape(x.shape[0], -1)  # flatten

        for l in self._fully_connected:
            x = nnx.relu(l(x))
        # x = mean(x, axis=(1, 2))
        # x = self.do1(nnx.relu(self.linear1(x)))
        # x = self.do2(nnx.relu(self.linear2(x)))
        x = self._head(x)
        return x

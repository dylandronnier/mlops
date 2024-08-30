from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jax import Array

from models._basic_cnn_block import BasicBlock


class _DenseLayer(nnx.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.norm1 = nnx.BatchNorm(num_features=num_input_features, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=num_input_features,
            out_features=bn_size * growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            rngs=rngs,
        )

        self.norm2 = nnx.BatchNorm(num_features=bn_size * growth_rate, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=bn_size * growth_rate,
            out_features=growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: list[Array], train: bool) -> Array:
        concated_features = jnp.concatenate(x, axis=-1)
        bottleneck_output = self.conv1(
            nnx.relu(self.norm1(concated_features, use_running_average=train))
        )
        return self.conv2(
            nnx.relu(self.norm2(bottleneck_output, use_running_average=train))
        )


class _DenseBlock(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.layers = list()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    rngs=rngs,
                )
            )

    def __call__(self, init_features: Array, train: bool) -> Array:
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features, train)
            features.append(new_features)
        return jnp.concat(features, axis=-1)


class _Transition(nnx.Module):
    def __init__(
        self, num_input_features: int, num_output_features: int, *, rngs: nnx.Rngs
    ) -> None:
        self.norm = nnx.BatchNorm(num_features=num_input_features, rngs=rngs)
        self.conv = nnx.Conv(
            in_features=num_input_features,
            out_features=num_output_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: Array, train: bool) -> Array:
        return nnx.avg_pool(
            self.conv(nnx.relu(self.norm(x, use_running_average=train))),
            window_shape=(2, 2),
            strides=(2, 2),
        )


@dataclass
class Architecture:
    channels: int = 3
    growth_rate: int = 12
    block_config: tuple[int, int, int, int] = (3, 6, 12, 8)
    num_init_features: int = 32
    bn_size: int = 4
    num_classes: int = 10


class NeuralNetwork(nnx.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.

    """

    def __init__(
        self,
        hp: Architecture,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.basic_cnn = BasicBlock(
            in_features=hp.channels, out_features=hp.num_init_features, rngs=rngs
        )

        self.layers = list()
        # Each denseblock
        num_features = hp.num_init_features
        for i, num_layers in enumerate(hp.block_config):
            self.layers.append(
                _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=hp.bn_size,
                    growth_rate=hp.growth_rate,
                    rngs=rngs,
                )
            )
            num_features = num_features + num_layers * hp.growth_rate
            if i != len(hp.block_config) - 1:
                self.layers.append(
                    _Transition(
                        num_input_features=num_features,
                        num_output_features=num_features // 2,
                        rngs=rngs,
                    )
                )
                num_features = num_features // 2

        # Final batch norm
        self.layers.append(nnx.BatchNorm(num_features=num_features, rngs=rngs))
        self.head = nnx.Linear(
            in_features=num_features, out_features=hp.num_classes, rngs=rngs
        )

    def __call__(self, x: Array, train: bool) -> Array:
        x = nnx.max_pool(self.basic_cnn(x, train), window_shape=(2, 2), strides=(2, 2))
        for layer in self.layers:
            x = layer(x, train)
        x = nnx.avg_pool(
            x, window_shape=(x.shape[1], x.shape[2]), strides=(x.shape[0], x.shape[1])
        )
        x = jnp.reshape(x, (x.shape[0], -1))
        return self.head(x)

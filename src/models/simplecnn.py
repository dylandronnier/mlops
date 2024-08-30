from dataclasses import dataclass

from flax import nnx
from jax import Array

from models._basic_cnn_block import BasicBlock


@dataclass
class Architecture:
    num_classes: int
    channels: int


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, arch: Architecture, *, rngs: nnx.Rngs):
        self.bb1 = BasicBlock(in_features=arch.channels, out_features=32, rngs=rngs)
        self.bb2 = BasicBlock(in_features=32, out_features=64, rngs=rngs)
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, arch.num_classes, rngs=rngs)

    def __call__(self, x: Array, train: bool) -> Array:
        x = nnx.avg_pool(self.bb1(x, train), window_shape=(2, 2), strides=(2, 2))
        x = nnx.avg_pool(self.bb2(x, train), window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x

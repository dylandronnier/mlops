import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike


@jax.jit
def squash(x: ArrayLike):
    """Squash activation function."""
    sq_norm = jnp.sum(jnp.square(x), -1, keepdims=True)
    return sq_norm * x / (1 + sq_norm) / jnp.sqrt(sq_norm + 1e-8)


class PrimaryCapsules(nnx.Module):
    """Primary Capsule."""

    def __init__(
        self,
        in_features: int,
        capsules: int,
        dim_caps: int,
        kernel_size: int,
        strides: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim_caps = dim_caps
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=capsules * dim_caps,
            kernel_size=(kernel_size, kernel_size),
            strides=strides,
            rngs=rngs,
        )

    def __call__(self, x):
        x = nnx.relu(self.conv(x))
        x = x.reshape((x.shape[0], -1, self.dim_caps))
        return squash(x)


class CapsNet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            in_features=1, out_features=256, kernel_size=(9, 9), rngs=rngs
        )
        self.primary_caps = PrimaryCapsules(
            in_features=256,
            capsules=32,
            dim_caps=8,
            kernel_size=9,
            strides=2,
            rngs=rngs,
        )

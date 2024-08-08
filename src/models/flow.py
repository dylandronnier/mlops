from typing import Sequence

import jax.numpy as jnp
from flax import nnx
from jax import random
from jax.lax import clamp
from jax.scipy.stats import norm


class ImageFlow(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()

    flows: Sequence[
        nnx.Module
    ]  # A list of flows (each a nn.Module) that should be applied on the images.
    import_samples: int = (
        8  # Number of importance samples to use during testing (see explanation below).
    )

    def __call__(self, x, testing=False):
        if not testing:
            bpd, rng = self._get_likelihood(x, rng)
        else:
            # Perform importance sampling during testing => estimate likelihood M times for each image
            img_ll, rng = self._get_likelihood(
                x.repeat(self.import_samples, 0), rng, return_ll=True
            )
            img_ll = img_ll.reshape(-1, self.import_samples)

            # To average the probabilities, we need to go from log-space to exp, and back to log.
            # Logsumexp provides us a stable implementation for this
            img_ll = nn.logsumexp(img_ll, axis=-1) - jnp.log(self.import_samples)

            # Calculate final bpd
            bpd = -img_ll * jnp.log2(jnp.exp(1)) / jnp.prod(x.shape[1:])
            bpd = bpd.mean()
        return bpd, rng

    def encode(self, imgs, rng):
        # Given a batch of images, return the latent representation z and
        # log-determinant jacobian (ldj) of the transformations
        z, ldj = imgs, jnp.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj, rng = flow(z, ldj, rng, reverse=False)
        return z, ldj, rng

    def _get_likelihood(self, imgs, rng, return_ll=False):
        """Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj, rng = self.encode(imgs, rng)
        log_pz = norm.logpdf(z).sum(axis=(1, 2, 3))
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * jnp.log2(jnp.exp(1)) / jnp.prod(imgs.shape[1:])
        return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        """Sample a batch of images from the flow."""
        # Sample latent representation from prior
        if z_init is None:
            rng, normal_rng = random.split(rng)
            z = random.normal(normal_rng, shape=img_shape)
        else:
            z = z_init

        # Transform z to x by inverting the flows
        # The log-determinant jacobian (ldj) is usually not of interest during sampling
        ldj = jnp.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj, rng = flow(z, ldj, rng, reverse=True)
        return z, rng


class Dequantization(nn.Module):
    alpha: float = 1e-5  # Small constant that is used to scale the original input for numerical stability.
    quants: int = (
        256  # Number of possible discrete values (usually 256 for 8-bit image)
    )

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, ldj, rng = self.dequant(z, ldj, rng)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += jnp.log(self.quants) * jnp.prod(z.shape[1:])
            z = jnp.floor(z)
            z = clamp(min=0.0, x=z, max=self.quants - 1.0).astype(jnp.int32)
        return z, ldj, rng

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z - 2 * nn.softplus(-z)).sum(axis=[1, 2, 3])
            z = nn.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= jnp.log(1 - self.alpha) * jnp.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = (
                z * (1 - self.alpha) + 0.5 * self.alpha
            )  # Scale to prevent boundaries 0 and 1
            ldj += jnp.log(1 - self.alpha) * jnp.prod(z.shape[1:])
            ldj += (-jnp.log(z) - jnp.log(1 - z)).sum(axis=[1, 2, 3])
            z = jnp.log(z) - jnp.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj, rng):
        # Transform discrete values to continuous volumes
        z = z.astype(jnp.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        ldj -= jnp.log(self.quants) * jnp.prod(z.shape[1:])
        return z, ldj, rng

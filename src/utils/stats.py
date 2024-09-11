from datasets import Dataset
from jax import Array
from jax.numpy import mean, std


def compute_mean_std(
    data: Dataset, axis=None, sample_size: int = 1_024
) -> tuple[Array, Array]:
    batch = next(data.iter(batch_size=sample_size, drop_last_batch=True))["image"]
    m = mean(batch, axis=axis)
    s = std(batch, axis=axis)
    return m, s

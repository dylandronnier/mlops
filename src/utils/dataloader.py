from typing import Union

from datasets import Dataset, DatasetDict
from jax import Array
from jax import numpy as jnp
from jax import random as jrand


def _epochiterator(data, batch_size: int, indices: Array):
    for i in range(0, len(indices), batch_size):
        idx = indices[i : i + batch_size]
        yield data[idx]


def _to_jax_dataset(dataset: Union[Dataset, DatasetDict]):
    return dataset.with_format("numpy")


class DataLoader:

    def __init__(
        self,
        dataset: Union[DatasetDict, Dataset],
        batch_size: int = 1,  # batch size
        shuffle: bool = False,  # if true, dataloader shuffles before sampling each batch
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.key = jrand.PRNGKey(seed)
        self.dataset = _to_jax_dataset(dataset)

        self.indices = jnp.arange(len(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def map(self, f):
        self.dataset = _to_jax_dataset(self.dataset.map(f))

    def __iter__(self):
        # shuffle (permutation) indices every epoch
        indices = (
            jrand.permutation(self.next_key(), self.indices)
            if self.shuffle
            else self.indices
        )

        if self.drop_last:
            indices = indices[: len(self.indices) - len(self.indices) % self.batch_size]
        return _epochiterator(self.dataset, self.batch_size, indices)

    def next_key(self):
        self.key, subkey = jrand.split(self.key)
        return subkey

    def __len__(self):
        complete_batches, remainder = divmod(len(self.indices), self.batch_size)
        return (
            complete_batches if self.drop_last else complete_batches + bool(remainder)
        )

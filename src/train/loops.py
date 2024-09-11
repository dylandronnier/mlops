from datasets import Dataset
from flax import nnx
from tqdm import tqdm

from .steps import eval_step, train_step


def train_loop(
    model: nnx.Module,
    dataset_train: Dataset,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch_size: int,
) -> None:
    model.train()
    for batch in tqdm(
        dataset_train.iter(batch_size=batch_size, drop_last_batch=True),
        desc="Training",
        total=len(dataset_train) // batch_size,
        leave=True,
    ):
        train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)


def eval_loop(
    model: nnx.Module,
    dataset_val: Dataset,
    metrics: nnx.MultiMetric,
    batch_size: int,
) -> None:
    model.eval()
    for batch in tqdm(
        dataset_val.iter(batch_size=batch_size, drop_last_batch=True),
        desc="Evaluating",
        total=len(dataset_val) // batch_size,
        leave=True,
    ):
        eval_step(model=model, metrics=metrics, batch=batch)

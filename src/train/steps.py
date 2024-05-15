from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import models
from clu import metrics
from datasets import DatasetDict
from flax import struct
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
from optax import sgd
from optax.losses import softmax_cross_entropy_with_integer_labels
from tqdm.auto import tqdm


@struct.dataclass
class Metrics(metrics.Collection):
    """Class that records the metrics through the training."""

    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


@dataclass
class ExperimentConfig:
    """Class that."""

    model: str
    epochs_number: int
    batch_size: int
    lr: float
    momentum: float


def create_train_state(rng, config: ExperimentConfig) -> TrainState:
    """Creates initial `TrainState`."""
    mod = getattr(models, config.model)()
    params = mod.init(rng, jnp.ones((16, 28, 28, 1)))["params"]
    return TrainState.create(
        apply_fn=mod.apply, params=params, tx=sgd(config.lr, config.momentum)
    )


def train_and_evaluate(
    state: TrainState, dataset: DatasetDict, batch_size: int
) -> tuple[TrainState, dict[str, dict[str, Any]]]:
    """2 loop for training and evaluation of the model."""
    summary_train = Metrics.empty()

    for batch in tqdm(
        dataset["train"].iter(batch_size=batch_size, drop_last_batch=True),
        desc="Training",
        total=len(dataset["train"]) // batch_size,
    ):
        state, summary_train = train_step(state, summary_train, batch)

    summary_eval = Metrics.empty()
    for batch in tqdm(
        dataset["test"].iter(batch_size=batch_size, drop_last_batch=True),
        desc=f"Evaluating:",
        total=len(dataset["test"]) // batch_size,
    ):
        summary_eval = eval_step(state, summary_eval, batch)

    return state, {"train": summary_train.compute(), "eval": summary_eval.compute()}


@jit
def train_step(state: TrainState, metrics: Metrics, batch):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = jnp.mean(
            softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["label"]
            )
        )
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = metrics.merge(
        Metrics.single_from_model_output(
            loss=loss, logits=logits, labels=batch["label"]
        )
    )
    return state, metrics


@jit
def eval_step(state: TrainState, metrics: Metrics, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    )
    metrics = metrics.merge(
        Metrics.single_from_model_output(
            loss=loss, logits=logits, labels=batch["label"]
        )
    )
    return metrics

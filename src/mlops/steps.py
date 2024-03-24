from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp
from clu import metrics
from datasets import DatasetDict
from flax import struct
from flax.training.train_state import TrainState
from jax import jit, value_and_grad
from optax import sgd
from optax.losses import softmax_cross_entropy_with_integer_labels
from tqdm.auto import tqdm

from mlops import models
from mlops.dist import make_config_suggest


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


@dataclass
class ExperimentConfig:
    model: str
    epochs_number: int
    batch_size: int
    lr: float
    momentum: float


Configsuggestion = make_config_suggest(ExperimentConfig)


def create_train_state(rng, config: ExperimentConfig) -> TrainState:
    """Creates initial `TrainState`."""
    mod = getattr(models, config.model)()
    params = mod.init(rng, jnp.ones((16, 28, 28, 1)))["params"]
    return TrainState.create(
        apply_fn=mod.apply, params=params, tx=sgd(config.lr, config.momentum)
    )


def train_and_evaluate(
    state: TrainState, dataset: DatasetDict, batch_size: int
) -> Tuple[TrainState, dict[str, dict[str, Any]]]:
    summary_train = Metrics.empty()

    for batch in tqdm(
        dataset["train"].iter(batch_size=batch_size, drop_last_batch=True),
        desc="Training",
        total=len(dataset["train"]) // batch_size,
    ):
        # print(batch["image"].shape)
        state, loss, logits = train_step(state, batch)
        summary_train = summary_train.merge(
            Metrics.single_from_model_output(
                loss=loss, logits=logits, labels=batch["label"]
            )
        )

    summary_eval = Metrics.empty()
    for batch in tqdm(
        dataset["test"].iter(batch_size=batch_size, drop_last_batch=True),
        desc=f"Evaluating:",
        total=len(dataset["test"]) // batch_size,
    ):
        loss, logits = eval_step(state, batch)
        summary_eval = summary_eval.merge(
            Metrics.single_from_model_output(
                loss=loss, logits=logits, labels=batch["label"]
            )
        )

    return state, {"train": summary_train.compute(), "eval": summary_eval.compute()}


@jit
def train_step(state: TrainState, batch):
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
    return state, loss, logits


@jit
def eval_step(state: TrainState, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = jnp.mean(
        softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"])
    )
    return loss, logits

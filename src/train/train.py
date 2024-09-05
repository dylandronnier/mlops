import logging
from dataclasses import dataclass

import mlflow
from datasets import DatasetDict
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from omegaconf import MISSING
from optax import sgd
from tqdm import tqdm

from train.steps import eval_step, train_step


@dataclass
class TrainingConfig:
    """Class that defines the parameters for the gradient descent."""

    # Number of epochs
    epochs: int = MISSING

    # Batch size
    batch_size: int = MISSING

    # Learning rate
    learning_rate: float = MISSING

    # Momentum.
    momentum: float = MISSING


def train_and_evaluate(
    model: nnx.Module,
    dataset: DatasetDict,
    training_config: TrainingConfig,
) -> nnx.Module:
    # Init the training state
    early_stop = EarlyStopping(patience=5, min_delta=1e-3)

    optimizer = nnx.Optimizer(
        model, sgd(training_config.learning_rate, training_config.momentum)
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    for epoch in range(1, training_config.epochs + 1):
        # Shuffle the dataset at each epoch
        dataset = dataset.shuffle(keep_in_memory=True)

        # Training loop
        model.train()
        for batch in tqdm(
            dataset["train"].iter(
                batch_size=training_config.batch_size, drop_last_batch=True
            ),
            desc="Training",
            total=len(dataset["train"]) // training_config.batch_size,
        ):
            train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

        # Log training metrics
        for metric, value in metrics.compute().items():  # compute metrics
            logging.info(f"Train {metric} = {value}")
            mlflow.log_metric(
                key=f"train_{metric}", value=float(value), step=epoch
            )  # record metrics
        metrics.reset()  # reset metrics for test set

        # Evaluation loop
        model.eval()
        for batch in tqdm(
            dataset["test"].iter(
                batch_size=training_config.batch_size, drop_last_batch=True
            ),
            desc="Evaluating",
            total=len(dataset["test"]) // training_config.batch_size,
        ):
            eval_step(model=model, metrics=metrics, batch=batch)
            # print(pred_step(model=model, batch=batch))
            # print(batch["label"])

        # Log test metrics
        for metric, value in metrics.compute().items():
            logging.info(f"Evaluation {metric} = {value}")
            if metric == "loss":
                early_stop = early_stop.update(value)
            mlflow.log_metric(key=f"test_{metric}", value=float(value), step=epoch)
        metrics.reset()  # reset metrics for next training epoch

        if early_stop.should_stop:
            logging.warning(
                "No improvments of the evaluation loss during"
                + f" the last {early_stop.patience} epochs."
            )
            logging.warning(f"Could not reach epoch {training_config.epochs}.")
            break

    logging.info(f"Best metric is equal to {early_stop.best_metric}")

    return model

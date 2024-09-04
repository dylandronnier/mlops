import logging
from dataclasses import asdict, dataclass

import mlflow
import numpy as np
from datasets import DatasetDict
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import tree_leaves
from matplotlib.pyplot import close
from omegaconf import MISSING
from optax import sgd
from tqdm import tqdm

from deploy.serve import FlaxModel
from train.steps import eval_step, pred_step, train_step
from utils.utils import show_img_grid


@dataclass
class TrainingConfig:
    """Class that defines the parameters for the gradient descent."""

    # Number of epochs
    epochs_number: int = MISSING

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
) -> float:
    # Log configuration parameters
    mlflow.log_params(asdict(training_config))

    # Init the training state
    early_stop = EarlyStopping(patience=3, min_delta=1e-3)

    optimizer = nnx.Optimizer(
        model, sgd(training_config.learning_rate, training_config.momentum)
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    mlflow.log_param(
        "nb_parameters", sum(p.size for p in tree_leaves(nnx.split(model)[1]))
    )

    for epoch in range(1, training_config.epochs_number + 1):
        # Training loop
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
        for batch in tqdm(
            dataset["test"].iter(
                batch_size=training_config.batch_size, drop_last_batch=True
            ),
            desc="Evaluating",
            total=len(dataset["test"]) // training_config.batch_size,
        ):
            eval_step(model=model, metrics=metrics, batch=batch)

        # Log test metrics
        for metric, value in metrics.compute().items():
            logging.info(f"Evaluation {metric} = {value}")
            if metric == "loss":
                early_stop = early_stop.update(value)
            mlflow.log_metric(key=f"test_{metric}", value=float(value), step=epoch)
        metrics.reset()  # reset metrics for next training epoch

        if early_stop.should_stop:
            logging.info("Stopping due to no improvments of the evaluation loss.")
            break

    # Inference testing of the model
    images = next(dataset["test"].iter(batch_size=10))
    fig = show_img_grid(images["image"], pred_step(model, images))
    mlflow.log_figure(
        figure=fig,
        artifact_file="inference.pdf",
    )
    close(fig)

    # Logging the model
    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=FlaxModel(*nnx.split(model)),
        input_example=np.array(images["image"]),  # TO CHANGE
        # registered_model_name="cnn",
    )
    return early_stop.best_metric

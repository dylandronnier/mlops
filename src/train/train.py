from dataclasses import asdict, dataclass

import mlflow
import numpy as np
from datasets import DatasetDict
from deploy.serve import FlaxModel
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import tree_leaves
from matplotlib.pyplot import close
from optax import sgd
from tqdm import tqdm
from utils.utils import show_img_grid

from train.steps import eval_step, pred_step, train_step


@dataclass
class TrainingConfig:
    """Class that."""

    # Number of epochs
    epochs_number: int

    # Batch size
    batch_size: int

    # Learning rate
    lr: float

    # Momentum.
    momentum: float


def train_and_evaluate(
    model: nnx.Module, dataset: DatasetDict, config: TrainingConfig
) -> float:
    # Log configuration parameters
    mlflow.log_params(asdict(config))

    # Log dataset
    # mlflow.log_input(
    #     from_huggingface(dataset["train"]),
    #     context="training",
    # )
    # mlflow.log_input(
    #     from_huggingface(dataset["test"], "test"),
    #     context="validation",
    # )

    # Init the training state

    early_stop = EarlyStopping(patience=3, min_delta=1e-3)

    optimizer = nnx.Optimizer(model, sgd(config.lr, config.momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    mlflow.log_param(
        "nb_parameters", sum(p.size for p in tree_leaves(nnx.split(model)[1]))
    )

    for epoch in range(1, config.epochs_number + 1):
        for batch in tqdm(
            dataset["train"].iter(batch_size=config.batch_size, drop_last_batch=True),
            desc="Training",
            total=len(dataset["train"]) // config.batch_size,
        ):
            train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

        # Log training metrics
        for metric, value in metrics.compute().items():  # compute metrics
            mlflow.log_metric(
                key=f"train_{metric}", value=float(value), step=epoch
            )  # record metrics
            metrics.reset()  # reset metrics for test set

        for batch in tqdm(
            dataset["test"].iter(batch_size=config.batch_size, drop_last_batch=True),
            desc="Evaluating:",
            total=len(dataset["test"]) // config.batch_size,
        ):
            eval_step(model=model, metrics=metrics, batch=batch)

            # Log test metrics
        for metric, value in metrics.compute().items():
            if metric == "loss":
                early_stop = early_stop.update(value)
            mlflow.log_metric(key=f"test_{metric}", value=float(value), step=epoch)
            metrics.reset()  # reset metrics for next training epoch

        if early_stop.should_stop:
            print("Stopping due to no improvments")
            break

    # Inference testing of the model
    images = next(dataset["test"].iter(batch_size=9))
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
        input_example=np.array(images["image"]),
        # registered_model_name="cnn",
    )
    return early_stop.best_metric

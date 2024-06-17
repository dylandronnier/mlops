from dataclasses import asdict, dataclass

import mlflow
import models
import numpy as np
import optuna
from datasets import DatasetDict
from deploy.serve import FlaxModel
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import tree_leaves
from matplotlib.pyplot import close
from mlflow.data.huggingface_dataset import from_huggingface
from optax import sgd
from tqdm import tqdm
from train.steps import eval_step, pred_step, train_step
from utils.dist import Distribution, make_config_suggest
from utils.utils import show_img_grid

RUN_ID_ATTRIBUTE_KEY = "mlflow_run_id"


@dataclass
class ExperimentConfig:
    """Class that."""

    model: str
    epochs_number: int
    batch_size: int
    lr: float
    momentum: float
    seed: int


Configsuggestion = make_config_suggest(ExperimentConfig)


def champion_callback(
    study: optuna.study.Study, trial: optuna.trial.FrozenTrial
) -> None:
    """Save the id of the best trial during the study."""
    if trial.value and trial.value <= study.best_value:
        study.set_user_attr(
            "winner_run_id", trial.system_attrs.get(RUN_ID_ATTRIBUTE_KEY)
        )


def objective(
    trial: optuna.trial.Trial,
    dataset: DatasetDict,
    dist_config: Distribution[ExperimentConfig],
) -> float:

    # Suggest config
    config = dist_config.suggest(trial=trial)

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

    model = models.CNN(rngs=nnx.Rngs(config.seed + trial.number))
    optimizer = nnx.Optimizer(model, sgd(config.lr, config.momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    mlflow.log_param(
        "nb_parameters", sum(p.size for p in tree_leaves(nnx.split(model)[1]))
    )

    # print(nnx.display(model))

    for epoch in range(1, config.epochs_number + 1):

        for batch in tqdm(
            dataset["train"].iter(batch_size=config.batch_size, drop_last_batch=True),
            desc="Training",
            total=len(dataset["train"]) // config.batch_size,
        ):
            train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

        # Log training metrics
        for metric, value in metrics.compute().items():  # compute metrics
            mlflow.log_metric(f"train_{metric}", value, step=epoch)  # record metrics
            metrics.reset()  # reset metrics for test set

        for batch in tqdm(
            dataset["test"].iter(batch_size=config.batch_size, drop_last_batch=True),
            desc=f"Evaluating:",
            total=len(dataset["test"]) // config.batch_size,
        ):
            eval_step(model=model, metrics=metrics, batch=batch)

            # Log test metrics
        for metric, value in metrics.compute().items():
            if metric == "loss":
                early_stop = early_stop.update(value)
            mlflow.log_metric(f"test_{metric}", value, step=epoch)
            metrics.reset()  # reset metrics for next training epoch

        if early_stop.should_stop:
            print("Stopping due to no improvments")
            break

    # Inference testing of the model
    images = next(dataset["test"].iter(batch_size=9))
    fig = show_img_grid(images["image"], pred_step(model, images))
    mlflow.log_figure(
        figure=fig,
        artifact_file=f"inference.pdf",
    )
    close(fig)

    # Logging the model
    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=FlaxModel(*nnx.split(model)),
        input_example=np.ones((1, 28, 28, 1)),
        # registered_model_name="cnn",
    )
    return early_stop.best_metric

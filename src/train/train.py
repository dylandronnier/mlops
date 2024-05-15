from dataclasses import asdict
from typing import TypeVar

import jax.numpy as jnp
import mlflow
import numpy as np
import optuna
from datasets import DatasetDict
from deploy.serve import FlaxModel
from flax.training.early_stopping import EarlyStopping
from jax import random, tree_map
from matplotlib.pyplot import close
from mlflow.data.huggingface_dataset import from_huggingface
from pandas.io.json._normalize import nested_to_record
from train.steps import (
    ExperimentConfig,
    create_train_state,
    train_and_evaluate,
)
from utils.dist import Distribution, make_config_suggest
from utils.utils import show_img_grid

X = TypeVar(name="X")

RUN_ID_ATTRIBUTE_KEY = "mlflow_run_id"


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
    rng = random.key(0)
    rng, init_rng = random.split(rng)
    state = create_train_state(init_rng, config)

    early_stop = EarlyStopping(patience=3, min_delta=1e-3)

    for epoch in range(1, config.epochs_number + 1):

        state, summary = train_and_evaluate(
            state=state, dataset=dataset, batch_size=config.batch_size
        )

        summary_dict = tree_map(float, nested_to_record(summary))

        mlflow.log_metrics(summary_dict)

        print(
            "Epoch %d Summary: train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,"
            " test_accuracy: %.2f" % (epoch, *summary_dict.values()),
            end="\n\n\n",
        )

        early_stop = early_stop.update(summary_dict["eval.loss"])

        if early_stop.should_stop:
            print("Stopping due to no improvments")
            break

    # Inference testing of the model
    images = next(dataset["test"].iter(batch_size=9))["image"]
    logits = state.apply_fn({"params": state.params}, images)
    fig = show_img_grid(images, jnp.argmax(logits, -1))
    mlflow.log_figure(
        figure=fig,
        artifact_file=f"inference.pdf",
    )
    close(fig)

    # Logging the model
    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=FlaxModel(state.params),
        input_example=np.ones((1, 28, 28, 1)),
        # registered_model_name="cnn",
    )
    return early_stop.best_metric

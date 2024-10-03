from typing import Any

import hydra
import jax.numpy as jnp
import mlflow
import optuna
from datasets import DatasetDict, load_dataset
from flax import nnx
from optuna.integration.mlflow import MLflowCallback

from train import TrainingConfig, train_and_evaluate
from utils.dist import RangeFloat, make_config_suggest


def preprocessing(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize the image dataset."""
    example["image"] = example["image"][..., jnp.newaxis] / 255
    return example


RUN_ID_ATTRIBUTE_KEY = "mlflow_run_id"


TrainingConfigSuggestion = make_config_suggest(TrainingConfig)


def champion_callback(
    study: optuna.study.Study, trial: optuna.trial.FrozenTrial
) -> None:
    """Save the id of the best trial during the study."""
    if trial.value and trial.value <= study.best_value:
        study.set_user_attr(
            "winner_run_id", trial.system_attrs.get(RUN_ID_ATTRIBUTE_KEY)
        )


@hydra.main
def main(num_trials: int, seed: int = 42) -> None:
    """Run hyperparameters search experiment with a basic CNN.

    Args:
    ----
      num_trials : number of trials in the research of hyperparameters.
      seed : seed of the experiment.

    """
    # Load the configuration search space
    dist_config = TrainingConfigSuggestion(
        epochs_number=4,
        batch_size=32,
        lr=RangeFloat(name="learning rate", low=5e-4, high=5e-2),
        momentum=RangeFloat(name="momentum", low=0.7, high=0.95),
    )

    # Load dataset
    dataset = load_dataset(
        path="ylecun/mnist",
        split={"train": "train[:5%]", "test": "test[:5%]"},
    )

    dataset = dataset.with_format("jax")

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(dataset, DatasetDict)):
        raise TypeError

    # Preprocess the data
    rescaled_dataset = dataset.map(preprocessing)

    # Enable system metrics logging by mlflow
    mlflow.enable_system_metrics_logging()

    # Optuna mlflow callback
    mlflc = MLflowCallback(metric_name="Loss", create_experiment=True)

    # New study for
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        sampler=sampler, direction="minimize", study_name="foox"
    )

    # Optimization
    study.optimize(
        mlflc.track_in_mlflow()(
            lambda trial: train_and_evaluate(
                CNN(rngs=nnx.Rngs(seed + trial.number)),
                rescaled_dataset,
                dist_config.suggest(trial),
            )
        ),
        n_trials=num_trials,
        timeout=2_000,
        callbacks=[mlflc, champion_callback],
    )
    best_run = study.user_attrs.get("winner_run_id")

    if best_run:
        artifact_uri = mlflow.get_run(best_run).info.artifact_uri
        if artifact_uri:
            mlflow.register_model(
                model_uri=artifact_uri + "/trained_model", name="cnn_nnx"
            )

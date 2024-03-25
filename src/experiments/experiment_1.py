from functools import partial
from typing import Any, Dict

import jax.numpy as jnp
import mlflow
from datasets import DatasetDict, load_dataset
from mlops.dist import RangeFloat
from mlops.steps import Configsuggestion
from mlops.train import champion_callback, objective
from optuna import create_study
from optuna.integration.mlflow import MLflowCallback


def preprocessing(example: Dict[str, Any]) -> Dict[str, Any]:
    example["image"] = example["image"][..., jnp.newaxis] / 255.0
    return example


if __name__ == "__main__":

    # Load the configuration search space
    config = Configsuggestion(
        model="CNN",
        epochs_number=2,
        batch_size=32,
        lr=RangeFloat(name="learning rate", low=5e-4, high=5e-2),
        momentum=RangeFloat(name="momentum", low=0.7, high=0.95),
    )

    # Load dataset
    dataset = load_dataset(path="rassibassi/sample_mnist")
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
    study = create_study(direction="minimize", study_name="new baz")

    # Optimization
    study.optimize(
        mlflc.track_in_mlflow()(
            partial(objective, dist_config=config, dataset=rescaled_dataset)
        ),
        n_trials=4,
        timeout=600,
        callbacks=[mlflc, champion_callback],
    )
    best_run = study.user_attrs.get("winner_run_id")
    artifact_uri = mlflow.get_run(best_run).info.artifact_uri
    if artifact_uri:
        mlflow.register_model(model_uri=artifact_uri + "/trained_model", name="cnn")

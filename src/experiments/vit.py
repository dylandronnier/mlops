from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import mlflow
import tyro
from datasets import DatasetDict, load_dataset
from flax import nnx
from models import VisionTransformer
from optuna.integration.mlflow import MLflowCallback
from train import TrainingConfig, train_and_evaluate


def preprocessing(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize the image dataset."""
    example["image"] = example["image"][..., jnp.newaxis] / 255
    return example


RUN_ID_ATTRIBUTE_KEY = "mlflow_run_id"


@dataclass
class ExperimentConfig:
    # training config
    training_config: TrainingConfig

    # Seed of the experiment
    seed: int


if __name__ == "__main__":
    # Fix parameters experiment through cli
    exp = tyro.cli(ExperimentConfig)

    # Load dataset
    dataset = load_dataset(
        path="uoft-cs/cifar10",
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

    model = VisionTransformer(
        embed_dim=256,
        mlp_dim=512,
        num_heads=8,
        layers=6,
        patches_size=4,
        num_patches=64,
        num_classes=10,
        dropout_rate=0.2,
        attendion_dropout_rate=0.2,
        rngs=nnx.Rngs(exp.seed),
    )

    train_and_evaluate(model, rescaled_dataset, exp.training_config)

import enum
from typing import Any

import mlflow
import tyro
from datasets import DatasetDict, load_dataset
from flax import nnx
from models import VisionTransformer
from models.densenet import DenseNet
from models.mlp_mixer import MlpMixer
from models.resnet import Resnet9
from train import TrainingConfig, train_and_evaluate


def preprocessing(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize the image dataset."""
    example["image"] = example["img"] / 255
    return example


class Model(enum.Enum):
    ViT = enum.auto()
    ResNet = enum.auto()
    MlpMixer = enum.auto()
    DenseNet = enum.auto()


@tyro.cli
def main(training_config: TrainingConfig, model: Model, seed: int = 42) -> None:
    """Train a VisionTransformer on the CIFAR10 dataset.

    Args:
      training_config: Hyperparameters of the training.
      model: Model architecture.
      seed: Seed of the experiment.

    """
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

    if model == Model.ResNet:
        mod = Resnet9(num_classes=10, rngs=nnx.Rngs(seed))
    elif model == Model.ViT:
        mod = VisionTransformer(
            embed_dim=256,
            mlp_dim=512,
            num_heads=8,
            layers=6,
            patches_size=4,
            num_patches=64,
            num_classes=10,
            dropout_rate=0.2,
            attendion_dropout_rate=0.2,
            rngs=nnx.Rngs(seed),
        )

    elif model == Model.MlpMixer:
        mod = MlpMixer(
            image_size=32,
            channels=3,
            num_classes=10,
            embed_dim=128,
            num_blocks=4,
            token_mlp_dim=16,
            channels_mlp_dim=128,
            patches_size=8,
            rngs=nnx.Rngs(seed),
        )
    elif model == Model.DenseNet:
        mod = DenseNet(rngs=nnx.Rngs(seed))

    train_and_evaluate(mod, rescaled_dataset, training_config)

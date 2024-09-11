from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from utils.confmodel import ModelConfig, store_model_config


@dataclass
class DatasetConfig:
    """Configuration of the dataset."""

    hf_id: str = MISSING
    images_width: int = MISSING
    images_height: int = MISSING
    channels: int = MISSING
    num_classes: int = MISSING
    image_column_name: str = "image"


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


@dataclass
class GlobalConfig:
    """Configuration of the experiment."""

    training_hp: TrainingConfig = MISSING
    model: ModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    seed: int = 42


def prepare_configuration_store(cs: ConfigStore):
    cs.store(name="base_config", node=GlobalConfig)
    cs.store(group="training_hp", name="base_trainingconfig", node=TrainingConfig)
    cs.store(group="dataset", name="base_datasetconfig", node=DatasetConfig)
    for module in ["visiontransformer", "densenet", "resnet", "simplecnn"]:
        store_model_config(cs=cs, module="models." + module)

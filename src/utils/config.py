import inspect
from abc import ABC
from dataclasses import dataclass, field, make_dataclass

from flax.nnx import Module
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from models import *
from models.densenet import DenseNet


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
class ModelConfig(ABC):
    _partial_: bool = True


def store_model_config(cs: ConfigStore, nn_cls: type[Module]) -> None:
    name = nn_cls.__name__
    mod = inspect.getmodule(nn_cls)
    if mod is None:
        raise TypeError
    module_name = mod.__name__

    new_fields = [("_target_", str, module_name + "." + name)]
    for f in inspect.signature(nn_cls.__init__).parameters.values():
        if f.name in ["channels", "num_classes", "rngs", "self"]:
            continue
        if f.default == inspect._empty:
            new_fields.append((f.name, f.annotation, field(default=MISSING)))
        else:
            new_fields.append((f.name, f.annotation, f.default))
    cls = make_dataclass(
        cls_name="Config" + name,
        fields=new_fields,
        bases=(ModelConfig,),
    )
    cs.store(group="model", name="base_" + name, node=cls)


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
    for cls in [SimpleCNN, ResNet, DenseNet, ViT]:
        store_model_config(cs, cls)

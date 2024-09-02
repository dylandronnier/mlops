from dataclasses import dataclass
from typing import Any

# import tyro
import hydra
import mlflow
from datasets import DatasetDict, load_dataset
from flax import nnx
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf, SCMode
from train import train_and_evaluate
from train.train import TrainingConfig
from utils.confmodel import ConfigModel, store_model_config
from utils.dataloader import DataLoader


def preprocessing(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize the image dataset."""
    example["image"] = example["image"] / 255
    return example


@dataclass
class Config:

    """Configuration of the experiment."""

    training_hp: TrainingConfig = MISSING
    model: ConfigModel = MISSING
    hf_dataset: str = "uoft-cs/cifar10"
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="training_hp", name="base_trainingconfig", node=TrainingConfig)
for module in ["visiontransformer", "densenet", "resnet"]:
    store_model_config(cs=cs, module="models." + module)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: Config) -> None:
    """Train a VisionTransformer on the CIFAR10 dataset.

    Args:
        conf(Config): Configuration of the experiment.

    """
    # Load dataset
    hf_dataset = load_dataset(
        path=conf.hf_dataset,
        # split={"train": "train[:5%]", "test": "test[:5%]"},
    ).rename_column("img", "image")

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        raise TypeError

    # Preprocess the data
    # rescaled_dataset = hf_dataset.map(preprocessing)

    dataloader_train = DataLoader(
        hf_dataset["train"],
        batch_size=conf.training_hp.batch_size,
        shuffle=True,
        drop_last=True,
    )

    dataloader_train.map(preprocessing)

    #
    dataloader_test = DataLoader(
        hf_dataset["test"],
        batch_size=conf.training_hp.batch_size,
        shuffle=True,
        drop_last=True,
    )

    dataloader_test.map(preprocessing)

    # Enable system metrics logging by mlflow
    mlflow.enable_system_metrics_logging()

    # Initialize the model
    dc_model = OmegaConf.to_container(
        cfg=conf.model, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    mod = dc_model.to_model(rngs=nnx.Rngs(conf.seed))

    # Train and evaluate
    dc_training_hp = OmegaConf.to_container(
        cfg=conf.training_hp, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )

    train_and_evaluate(mod, dataloader_train, dataloader_test, dc_training_hp)


if __name__ == "__main__":
    main()

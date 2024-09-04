from dataclasses import dataclass
from functools import partial

import augmax
import hydra
import jax.numpy as jnp
import jax.random as jrand
import mlflow
from datasets import Array3D, DatasetDict, load_dataset
from flax import nnx
from hydra.core.config_store import ConfigStore
from jax import jit
from omegaconf import MISSING, OmegaConf, SCMode

from train import train_and_evaluate
from train.train import TrainingConfig
from utils.confmodel import ConfigModel, store_model_config

# train_transforms = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
#     ]
# )

mean = jnp.array([0.4914, 0.4822, 0.4465])
std = jnp.array([0.203, 0.1994, 0.2010])
pre_transform = augmax.Chain(
    augmax.ByteToFloat(),
    augmax.Normalize(mean=mean, std=std),
)
key = jrand.PRNGKey(0)
transform = jit(partial(pre_transform, rng=key))


@dataclass
class Config:
    """Configuration of the experiment."""

    training_hp: TrainingConfig = MISSING
    model: ConfigModel = MISSING
    hf_dataset: str = "cifar10"
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="training_hp", name="base_trainingconfig", node=TrainingConfig)
for module in ["visiontransformer", "densenet", "resnet"]:
    store_model_config(cs=cs, module="models." + module)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(conf: Config) -> None:
    """Train a VisionTransformer on the CIFAR10 dataset.

    Args:
        conf(Config): Configuration of the experiment.

    """
    # Load dataset
    hf_dataset = load_dataset(
        path=conf.hf_dataset,
    ).rename_column("img", "image")

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        raise TypeError

    # Preprocess the data

    # transform = jit(partial(augmax.ByteToFloat().pixelwise, rng=key))
    hf_dataset = hf_dataset.with_format("jax")
    hf_dataset = hf_dataset.map(
        lambda ex: {"image": transform(inputs=ex["image"])},
        batched=True,
        batch_size=16,
    )
    # hf_dataset = hf_dataset.map(preprocessing)
    hf_dataset = hf_dataset.cast_column(
        "image", feature=Array3D(shape=(32, 32, 3), dtype="float32")
    )

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

    train_and_evaluate(mod, hf_dataset, dc_training_hp)


if __name__ == "__main__":
    app()

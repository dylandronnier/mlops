import logging
from dataclasses import asdict, dataclass
from functools import partial

import augmax
import hydra
import jax.numpy as jnp
import mlflow
import numpy as np
from datasets import Array3D, DatasetDict, load_dataset
from flax import nnx
from hydra.core.config_store import ConfigStore
from jax import jit, vmap
from matplotlib.pyplot import close
from omegaconf import MISSING, OmegaConf, SCMode

from deploy.serve import FlaxModel
from train import train_and_evaluate
from train.steps import pred_step
from train.train import TrainingConfig
from utils.confmodel import ModelConfig, store_model_config
from utils.networks import number_of_parameters
from utils.plot import show_img_grid

mean = jnp.array([0.4914, 0.4822, 0.4465])
std = jnp.array([0.203, 0.1994, 0.2010])
transform = augmax.Chain(
    augmax.ByteToFloat(),
    augmax.Normalize(mean=mean, std=std),
)
jit_transform = jit(partial(transform, rng=jnp.array(0)))


def plot_inference(model: nnx.Module, batch):
    inputs = vmap(partial(transform, rng=jnp.array(0)))(batch["image"])
    predicted_label = pred_step(model, inputs)
    return show_img_grid(batch["image"], predicted_label)


@dataclass
class Config:
    """Configuration of the experiment."""

    training_hp: TrainingConfig = MISSING
    model: ModelConfig = MISSING
    hf_dataset: str = "cifar10"
    seed: int = 42


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="training_hp", name="base_trainingconfig", node=TrainingConfig)
for module in ["visiontransformer", "densenet", "resnet", "simplecnn"]:
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
    hf_dataset_gpu = (
        hf_dataset.with_format("jax")
        .map(
            lambda ex: {"image": jit_transform(inputs=ex["image"])},
            batched=True,
            batch_size=16,
        )
        .cast_column("image", feature=Array3D(shape=(32, 32, 3), dtype="float32"))
    )

    logging.info(
        msg=f"The dataset {conf.hf_dataset} has been preprocessed. "
        + "It is ready for the learning task."
    )

    # Enable system metrics logging by mlflow
    mlflow.enable_system_metrics_logging()

    # Initialize the model
    dc_model = OmegaConf.to_container(
        cfg=conf.model, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    mod = dc_model.to_model(rngs=nnx.Rngs(conf.seed))
    mlflow.log_param("nb_parameters", number_of_parameters(mod))

    # Train and evaluate
    dc_training_hp = OmegaConf.to_container(
        cfg=conf.training_hp, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    # Log configuration parameters
    mlflow.log_params(asdict(dc_training_hp))

    mod = train_and_evaluate(mod, hf_dataset_gpu, dc_training_hp)

    # Inference testing of the model
    batch = next(hf_dataset["test"].iter(batch_size=10))
    fig = plot_inference(mod, batch)
    mlflow.log_figure(
        figure=fig,
        artifact_file="inference.pdf",
    )
    close(fig)

    # Logging the model
    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=FlaxModel(*nnx.split(mod)),
        input_example=np.array(batch["image"]),  # TO CHANGE
        # registered_model_name="cnn",
    )


if __name__ == "__main__":
    app()

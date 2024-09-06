import logging
from dataclasses import asdict
from functools import partial

import augmax
import hydra
import jax.numpy as jnp
import jax.random as jrand
import mlflow
import numpy as np
from datasets import Array3D, DatasetDict, load_dataset
from flax import nnx
from hydra.core.config_store import ConfigStore
from jax import jit
from omegaconf import OmegaConf, SCMode

from deploy.serve import FlaxModel
from train import train_and_evaluate
from train.steps import pred_step
from utils.config import GlobalConfig, prepare_configuration_store
from utils.networks import number_of_parameters

cs = ConfigStore.instance()
prepare_configuration_store(cs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(conf: GlobalConfig) -> None:
    """Train a VisionTransformer on the CIFAR10 dataset.

    Args:
        conf(Config): Configuration of the experiment.

    """
    # Load dataset
    hf_dataset = load_dataset(
        path=conf.dataset.hf_id,
    )

    if conf.dataset.image_column_name != "image":
        hf_dataset = hf_dataset.rename_column(conf.dataset.image_column_name, "image")

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        raise TypeError

    if conf.dataset.channels == 3:
        mean = jnp.array([0.4914, 0.4822, 0.4465])
        std = jnp.array([0.203, 0.1994, 0.2010])
    else:
        mean = jnp.array(0.5)
        std = jnp.array(0.2)

    transform = augmax.Chain(
        augmax.ByteToFloat(),
        augmax.Normalize(mean=mean, std=std),
    )
    jit_transform = jit(partial(transform, rng=jrand.PRNGKey(conf.seed)))

    # Preprocess the data
    hf_dataset_gpu = hf_dataset.with_format("jax")
    hf_dataset_gpu = hf_dataset_gpu.map(
        lambda ex: {"image": jit_transform(inputs=jnp.expand_dims(ex["image"], -1))},
        batched=True,
        batch_size=16,
    )

    hf_dataset_gpu = hf_dataset_gpu.cast_column(
        "image",
        feature=Array3D(
            shape=(
                conf.dataset.images_width,
                conf.dataset.images_height,
                conf.dataset.channels,
            ),
            dtype="float32",
        ),
    )

    logging.info(
        msg=f"The dataset {conf.dataset.hf_id} has been preprocessed. "
        + "It is ready for the learning task."
    )

    # Enable system metrics logging by mlflow
    mlflow.enable_system_metrics_logging()

    # Initialize the model
    dc_model = OmegaConf.to_container(
        cfg=conf.model, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )
    mod = dc_model.to_model(
        channels=conf.dataset.channels,
        num_classes=conf.dataset.num_classes,
        rngs=nnx.Rngs(conf.seed),
    )
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
    inputs = jit_transform(inputs=jnp.expand_dims(np.array(batch["image"]), -1))
    predicted_label = pred_step(mod, inputs)
    for im, pred_l, true_l in zip(batch["image"], predicted_label, batch["label"]):
        mlflow.log_image(im, key=f"Predicted {pred_l} / True {true_l}")

    # Logging the model
    mlflow.pyfunc.log_model(
        artifact_path="trained_model",
        python_model=FlaxModel(*nnx.split(mod)),
        input_example=np.array(batch["image"]),  # TO CHANGE
        # registered_model_name="cnn",
    )


if __name__ == "__main__":
    app()

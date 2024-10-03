import logging
from functools import partial

import augmax
import hydra
import jax.numpy as jnp
import jax.random as jrand
from aim import Image, Run
from aim.hf_dataset import HFDataset
from datasets import Array3D, DatasetDict, load_dataset
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from jax import jit
from optax import sgd

from train import eval_loop, log_and_track_metrics, pred_step, train_loop
from utils.config import (
    GlobalConfig,
    prepare_configuration_store,
)
from utils.networks import number_of_parameters
from utils.stats import compute_mean_std

# Setup Hydra
cs = ConfigStore.instance()
prepare_configuration_store(cs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(conf: GlobalConfig) -> None:
    """Train a VisionTransformer on the CIFAR10 dataset.

    Args:
        conf (Config): Configuration of the experiment.

    """
    # Start recording with Aim
    run = Run(system_tracking_interval=10, log_system_params=True)

    # Load dataset
    hf_dataset = load_dataset(
        path=conf.dataset.hf_id,
    )

    # Ensure the datasets is a Dataset Dictionary
    if not (isinstance(hf_dataset, DatasetDict)):
        logging.error(msg="Dataset is not splitted in test and train.")
        return

    run["datasets_info"] = HFDataset(hf_dataset)

    # Rename the image column with the proper name
    if conf.dataset.image_column_name != "image":
        hf_dataset = hf_dataset.rename_column(conf.dataset.image_column_name, "image")

    # Load the data in the GPU
    hf_dataset_gpu = hf_dataset.with_format("jax")
    hf_dataset_gpu = hf_dataset_gpu.shuffle(seed=conf.seed, keep_in_memory=True)

    # Compute mean & std of the dataset per channel
    mean, std = compute_mean_std(hf_dataset_gpu["train"], axis=(0, 1, 2))

    # Normalize data
    transform = augmax.Chain(
        augmax.Normalize(mean=mean, std=std),
        augmax.ByteToFloat(),
    )
    jit_transform = jit(partial(transform, rng=jrand.PRNGKey(conf.seed)))
    hf_dataset_gpu = hf_dataset_gpu.map(
        lambda ex: {"image": jit_transform(inputs=jnp.expand_dims(ex["image"], -1))},
        batched=True,
        batch_size=16,
    )

    # Recognize the gpu
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

    # Initialize the model
    mod = instantiate(conf.model)(
        num_classes=conf.dataset.num_classes,
        channels=conf.dataset.channels,
        rngs=nnx.Rngs(conf.seed),
    )

    # Save model architecture in AIM
    run["model"] = {"nb_parameters": number_of_parameters(mod)}

    # Train and evaluate
    # Log configuration parameters
    run["hparams"] = conf.training_hp

    # Init the training state
    early_stop = EarlyStopping(patience=5, min_delta=1e-3)

    optimizer = nnx.Optimizer(
        mod, sgd(conf.training_hp.learning_rate, conf.training_hp.momentum)
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    for epoch in range(1, conf.training_hp.epochs + 1):
        # Shuffle the dataset at each epoch
        hf_dataset_gpu["train"] = hf_dataset_gpu["train"].shuffle(
            seed=conf.seed, keep_in_memory=True
        )

        # Training loop
        train_loop(
            mod,
            hf_dataset_gpu["train"],
            optimizer,
            metrics,
            conf.training_hp.batch_size,
        )

        # Log training metrics
        log_and_track_metrics(metrics, subset="Train", run=run, epoch=epoch)

        # Reset metrics for test set
        metrics.reset()

        # Evaluation loop
        eval_loop(mod, hf_dataset_gpu["test"], metrics, conf.training_hp.batch_size)

        # Log test metrics
        log_and_track_metrics(metrics, subset="Validation", run=run, epoch=epoch)

        early_stop = early_stop.update(metrics.loss.compute())
        metrics.reset()  # reset metrics for next training epoch

        if early_stop.should_stop:
            logging.warning(
                "No improvments of the evaluation loss during"
                + f" the last {early_stop.patience} epochs."
            )
            logging.warning(f"Could not reach epoch {conf.training_hp.epochs}.")
            break

        if epoch % 2 == 0:
            predicted_label = pred_step(
                mod, next(hf_dataset_gpu["test"].iter(batch_size=20))["image"]
            )
            for i, (b, pred_l) in enumerate(
                zip(hf_dataset["test"].iter(batch_size=1), predicted_label)
            ):
                # run.log_image(im, key=f"Predicted {pred_l} / True {true_l}")
                run.track(
                    Image(b["image"][0]),
                    name=f"Batch {i}",
                    epoch=epoch,
                    context={"true": str(b["label"][0]), "prediction": str(pred_l)},
                )

    logging.info(f"Best metric is equal to {early_stop.best_metric}")

    # Logging the model
    # mlflow.pyfunc.log_model(
    #     artifact_path="trained_model",
    #     python_model=FlaxModel(*nnx.split(mod)),
    #     input_example=np.array(batch["image"]),  # TO CHANGE
    #     # registered_model_name="cnn",
    # )


if __name__ == "__main__":
    app()

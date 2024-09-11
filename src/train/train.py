import logging

from aim import Run
from flax import nnx


def log_and_track_metrics(
    metrics: nnx.MultiMetric, subset: str, run: Run, epoch: int
) -> None:
    # Log training metrics
    for metric, value in metrics.compute().items():  # compute metrics
        logging.info(f"{subset}: {metric} = {value} at epoch {epoch}.")
        run.track(value=value, name=metric, context={"subset": subset}, epoch=epoch)

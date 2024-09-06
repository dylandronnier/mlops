from flax import nnx
from jax import Array
from optax.losses import softmax_cross_entropy_with_integer_labels


def loss_fn(model: nnx.Module, batch):
    """Cross entropy losss function."""
    logits = model(batch["image"])
    loss = softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model=model, batch=batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(grads)


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model=model, batch=batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


@nnx.jit
def pred_step(model: nnx.Module, images: Array):
    logits = model(images)
    return logits.argmax(axis=1)

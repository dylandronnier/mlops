from flax import nnx
from jax.tree_util import tree_leaves


def number_of_parameters(mod: nnx.Module) -> int:
    """Compute the number of parameters in the model."""
    return sum(p.size for p in tree_leaves(nnx.split(mod)[1]))

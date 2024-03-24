from typing import Optional

import numpy as np
from flax.core import FrozenDict
from flax.serialization import from_bytes
from jax.nn import softmax
from jax.random import key
from mlflow.pyfunc import PythonModel, PythonModelContext

from mlops.models.cnn import CNN


class FlaxModel(PythonModel):
    def __init__(self, params: Optional[FrozenDict] = None) -> None:

        self._cnn = CNN()
        rng = key(0)
        self._params = self._cnn.init(rng, np.ones([1, 28, 28, 1]))["params"]

        if params:
            self._params = params
            self._init = True
        else:
            self._init = False

    def load_context(self, context: PythonModelContext) -> None:
        if not self._init:
            self._params = from_bytes(self._params, context.artifacts["weights"])

    def predict(self, context: PythonModelContext, model_input):
        return softmax(self._cnn.apply({"params": self._params}, model_input))

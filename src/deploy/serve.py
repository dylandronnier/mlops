from typing import Optional

import numpy as np
from flax import nnx
from flax.serialization import from_bytes
from jax.nn import softmax
from mlflow.pyfunc import PythonModel, PythonModelContext

from train.steps import pred_step


class FlaxModel(PythonModel):
    def __init__(self, graphdef: nnx.GraphDef, state: Optional[nnx.State]) -> None:
        self._graphdef = graphdef
        if state:
            self._model = nnx.merge(self._graphdef, state)
            self._model.eval()
            self._init = True
        else:
            self._init = False

    def load_context(self, context: PythonModelContext) -> None:
        if not self._init:
            self._state = from_bytes(self._state, context.artifacts["weights"])

    def predict(self, context: PythonModelContext, input) -> np.ndarray:
        logits = pred_step(self._model, input)
        return np.array(softmax(logits))

import numpy as np
from typing import Union
from ..models import Model
from ..utils import bitstring_to_state
from ._evaluator import LoschmidtEchoEvaluator

class ExactLoschmidtEcho(LoschmidtEchoEvaluator):
    """Class for evaluating Loschmidt echos of a model exactly.

    The error estimates of this evaluator are always zero.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamilotonain.
    """

    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, psi_bistring: Union[list, np.ndarray], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inherited method. See :py:meth:`LoschmidtEchoEvaluator.evaluate`."""
        psi = bitstring_to_state(psi_bistring)
        G = np.zeros(t.size, dtype=np.cdouble)
        G_err = np.zeros(t.size, dtype=np.cdouble)
        for i in range(t.size):
            G[i] = np.vdot(psi, self.model.evolve_state_exact(psi, t[i]))
            G_err[i] = 0.0 + 1j * 0.0
        return G, G_err

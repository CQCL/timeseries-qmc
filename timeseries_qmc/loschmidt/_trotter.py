from typing import Union
import numpy as np
import math
from ..models import Model
from ..utils import bitstring_to_state
from ._evaluator import LoschmidtEchoEvaluator

class TrotterizedLoschmidtEcho(LoschmidtEchoEvaluator):
    """Class for evaluating Loschmidt echos of a model via Trotterization.

    The error estimates of this evaluator are rough estimates of Trotterization error.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamilotonain.

    trotter_order: int
        The order of Tortterization. Currently, only 1st and 2nd orders are supported.
        
    n_steps: int
        Number of Trotter steps. Either this parameter or **dt_trotter** needs to be specified.
        Both parameters cannot be specified simultaneously.

    dt_trotter: float
        Size of Trotter step. Either this parameter or **n_steps** needs to be specified.
        Both parameters cannot be specified simultaneously.

    err_factor: float
        Prefactor for error estimates :math:`C`.

    err_scaling: str
        Specifies how the Trotter errors are estiamted. The value can be either 'linear' or 'quadratic'.
        Linear scaling estimates the error as :math:`C t^2/n`, while quadratic scaling estimates it as
        :math:`C t^3/n^2`, where :math:`n` is the number of Trotter steps and :math:`t` is the total evolution time.

        .. note:: Generally 1st-order Trotter has linear scaling while 2nd-order Trotterization has quadratic scaling.
                  However, in some special cases, even 1st-order Trotter has quadratic scaling.
    """

    def __init__(
        self,
        model: Model,
        trotter_order: int = 1,
        n_steps: int = None,
        dt_trotter: float = None,
        err_factor: float = 0,
        err_scaling: str = "quadratic",
    ):
        self.model = model
        self.trotter_order = trotter_order
        self.n_steps = n_steps
        self.dt_trotter = dt_trotter
        if dt_trotter is None and n_steps is None:
            raise ValueError("Either dt_trotter or n_steps should be specified")
        if dt_trotter is not None and n_steps is not None:
            raise ValueError("Only dt_trotter or n_steps can be specified")

        self.err_factor = err_factor
        self.err_scaling = err_scaling

    def evaluate(self, psi_bistring: Union[list, np.ndarray], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inherited method. See :py:meth:`LoschmidtEchoEvaluator.evaluate`."""
        psi = bitstring_to_state(psi_bistring)
        G = np.zeros(t.size, dtype=np.cdouble)
        G_err = np.zeros(t.size, dtype=np.cdouble)
        for i in range(t.size):
            G[i], G_err[i] = self._evaluate_timepoint(psi, t[i])
        return G, G_err

    def _evaluate_timepoint(self, psi_bistring, t):
        if self.n_steps is not None:
            n_steps = self.n_steps
        else:
            n_steps = math.ceil(abs(t) / self.dt_trotter)
            n_steps = max(1, n_steps)

        if self.err_scaling == "linear":
            err = self.err_factor * (1 + 1j) * t**2 / n_steps
        elif self.err_scaling == "quadratic":
            err = self.err_factor * (1 + 1j) * t**3 / n_steps**2

        if self.trotter_order == 1:
            return np.vdot(psi_bistring, self.model.evolve_state_1st_trotter(psi_bistring, t, n_steps)), err
        elif self.trotter_order == 2:
            return np.vdot(psi_bistring, self.model.evolve_state_2nd_trotter(psi_bistring, t, n_steps)), err

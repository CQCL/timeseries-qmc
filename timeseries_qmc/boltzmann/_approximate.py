from typing import Union
import math
import numpy as np
from ..models import Model
from ..utils import bitstring_to_state
from ._calculator import BoltzmannWeightCalculator

class ApproximateBoltzmannWeight(BoltzmannWeightCalculator):
    """Class for calculating Botlzmann weights of a model approximately by assuming that the local density of states is
    a Gaussian centered at the energy of the state and whose varaince equals the energy variance of the state.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamilotonain.

    beta: float
        inverse temperature.
    """

    def __init__(self, model: Model, beta: float):
        self.model = model
        self.beta = beta

    def calculate(self, psi_bitstring: Union[list, np.ndarray]) -> tuple[float, float, float]:
        """Inherited method. See :py:meth:`BoltzmannWeightCalculator.calculate`."""
        psi = bitstring_to_state(psi_bitstring)
        mean = self.model.calc_energy_mean(psi)
        var = self.model.calc_energy_variance(psi)
        W = math.exp(-self.beta * mean)
        C = self.beta * var

        W = W * np.exp(C * self.beta / 2)
        local_e = mean - C
        local_e2 = mean**2 - 2 * C * mean + C**2 + var
        return W, local_e, local_e2

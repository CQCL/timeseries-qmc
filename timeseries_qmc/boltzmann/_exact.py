from typing import Union
import numpy as np
from ..models import Model
from ..utils import bitstring_to_state
from ._calculator import BoltzmannWeightCalculator


class ExactBoltzmannWeight(BoltzmannWeightCalculator):
    """Class for calculating Botlzmann weights of a model exactly.
    
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

    def calculate(self, psi_bistring: Union[list, np.ndarray]) -> tuple[float, float, float]:
        """Inherited method. See :py:meth:`BoltzmannWeightCalculator.calculate`."""
        return self.model.calc_boltzmann_weight(bitstring_to_state(psi_bistring), self.beta)

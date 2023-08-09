from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class BoltzmannWeightCalculator(ABC):
    """Abstract base class for calculating Botlzmann weights and the associated local energy moments."""

    @abstractmethod
    def calculate(self, psi_bitstring: Union[list, np.ndarray]) -> tuple[float, float, float]:
        r"""Calculate the Boltzmann weight of a state and the first and second moments of its local energy.

        Parameters
        ----------
        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing some product state :math:`|\psi\rangle`.

        Returns
        -------
        w: float
            Boltzmann weight :math:`\langle\psi| e^{-\beta H}|\psi \rangle`.

        e: float
            Local value of energy :math:`{\langle\psi| e^{-\beta H} H |\psi \rangle}/
            {\langle\psi| e^{\beta H}|\psi \rangle}`.

        e_sq: float
            Local value of energy's second moment :math:`{\langle\psi| e^{-\beta H} H^2 |\psi \rangle}/
            {\langle\psi| e^{\beta H}|\psi \rangle}`.
        """
        raise NotImplementedError()

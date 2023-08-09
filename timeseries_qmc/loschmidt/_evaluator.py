from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class LoschmidtEchoEvaluator(ABC):
    """Abstract base class for evaluating Loschmidt echos."""

    @abstractmethod
    def evaluate(self, psi_bistring: Union[list, np.ndarray], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Evaluate Loschmidt echos and estimate their errors for the specified state and times.

        Parameter
        ---------
        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing some product state :math:`|\psi\rangle`.

        t:  :py:class:`numpy.ndarray`
            Array of time points :math:`t_i` at which to evluate Loschmidt echos.
            
        Returns
        -------
        echos: :py:class:`numpy.ndarray`
            Array of Loschmidt echos :math:`\langle\psi|e^{-H t_i}|\psi\rangle` at time points :math:`t_i`.
            Note that Loschmidt echos are complex numbers.

        errors: :py:class:`numpy.ndarray`
            Array of estimates of the errors made in computing `echos`.
            Note that the elements of the array are complex numbers. The real/imaginary part represents the error
            in the real/imaginary part of the echo, respectively.
        """
        raise NotImplementedError()

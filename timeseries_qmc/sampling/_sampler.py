from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Sampler(ABC):
    """Abstract base class for sampling quantum states."""

    @abstractmethod
    def propose_sample(self, old_sample_bitstring: Union[list, np.ndarray], rng: np.random.RandomState):
        r"""Proposes a new random sample based on the old one.

        Parameters
        ----------
        old_sample_bitstring: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing old product state :math:`|\psi\rangle`.            

        rng: :class:`numpy.random.RandomState`
            Random number generator.

        Returns
        -------
        new_sample_bitstring: :py:class:`numpy.ndarray`
            List/array of zeros and ones representing new product state  :math:`\psi^\prime`.

        proposal_ratio: float
            Ratio of proposal probabilities :math:`\frac{p(\psi^\prime \to \psi)}{p(\psi \to \psi^\prime )}`
        """
        raise NotImplementedError()

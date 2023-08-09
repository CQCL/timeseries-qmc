from typing import Union
import math
import numpy as np
from ..loschmidt import LoschmidtEchoEvaluator
from ._calculator import BoltzmannWeightCalculator
from ._numerics import filter_timeseries, fourier_transform, simple_truncation, symmetrize_timeseries, integrate_density


class FourierTransform(BoltzmannWeightCalculator):
    """Class for calculating Botlzmann weights from time series of Loschmidt echo via direct Fourier trasnfrom.

    Parameters
    ----------
    beta: float
        Inverse temperature.

    loschmidt_echo_evaluator: :class:`.loschmidt.LoschmidtEchoEvaluator`
        Evaluator of Loschimidt echos used to calculate the time series.

    t_max: float
        Maximum time used to evaluate Loschmidt echos.

    n_t: int
        Number of time points used to evaluate Loschmidt echos.

    n_omega: int
        Number of points of the frequency grid used to recostruct the local density of states.

    omega_max: float
        Maximum frequency of the grid used to reconstruct the local density of states.
        If ``None`` is used, Nyquist frequency is used based on the specified time parameters.

    delta: float
        Inverse of the width of the Gaussian filter used on the time series.
        Note that this Gaussian filter effectively broadens the density of states by `delta`.
        If ``None`` is used, the reciprocal of the maximum time is used.

    cutoff_factor: float
        Parameter :math:`C` controlling the cutoff on the density of states.
        The actual cutoff is :math:`C` times the maximum absolute value of the negative part of the density.
    """

    def __init__(
        self,
        beta: float,
        loschmidt_echo_evaluator: LoschmidtEchoEvaluator,
        t_max: float,
        n_t: int,
        n_omega: int = 1024,
        omega_max: float = None,
        delta: float = None,
        cutoff_factor: float = 2,
    ):
        self.loschmidt_echo_evaluator = loschmidt_echo_evaluator
        self.beta = beta
        self.n_t = n_t
        self.t_max = t_max
        self.dt = t_max / n_t
        self.t = np.linspace(self.dt, t_max, n_t, endpoint=True)
        if omega_max is None:
            omega_max = math.pi / self.dt
        self.n_omega = n_omega
        self.omega_max = omega_max
        self.omega, self.d_omega = np.linspace(-omega_max, omega_max, n_omega, endpoint=False, retstep=True)
        if delta is None:
            delta = 1.0 / t_max
        self.delta = delta
        self.cutoff_factor = cutoff_factor

    def estimate_density(self, psi_bitstring: Union[list, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        r"""Esimate the local density of states.

        Parameters
        ----------
        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing some product state :math:`|\psi\rangle`.

        Returns
        -------
        D: :py:class:`numpy.ndarray`
            Local density of state.

        omega: :py:class:`numpy.ndarray`
            Grid points on which the local density of states is evaluated.

        d_omega: float
            Grid spacing of **omega**.
        """
        G, G_err = self.loschmidt_echo_evaluator.evaluate(psi_bitstring, self.t)
        G = filter_timeseries(G, self.t, self.delta)
        G_symmetric, t_symmetric = symmetrize_timeseries(G, self.t)
        D = fourier_transform(G_symmetric, t_symmetric, self.omega)

        cutoff = self.cutoff_factor * np.abs(np.min(D))
        D, omega = simple_truncation(D, self.omega, cutoff)
        # Adjust normalization
        D = D / np.sum(D * self.d_omega)
        return D, omega, self.d_omega

    def calculate(self, psi_bitstring: list) -> tuple[float, float, float]:
        """Inherited method. See :py:meth:`BoltzmannWeightCalculator.calculate`."""
        D, omega, d_omega = self.estimate_density(psi_bitstring)
        return integrate_density(D, omega, d_omega, self.beta, self.delta)

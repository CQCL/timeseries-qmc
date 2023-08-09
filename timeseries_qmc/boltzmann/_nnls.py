from typing import Union
import math
import numpy as np
from ..models import Model
from ..loschmidt import LoschmidtEchoEvaluator
from ..utils import bitstring_to_state
from ._calculator import BoltzmannWeightCalculator
from ._numerics import adjust_moments, filter_timeseries, non_negative_least_squares_filtered, integrate_density

ERR_EPSILLON = 1e-6

class NNLS(BoltzmannWeightCalculator):
    r"""Class for calculating Botlzmann weights from time series of Loschmidt echo via non-negative least squares.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamiltonain.

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

    delta_factor: float
        Parameter :math:`C_\delta` used to determine the inverse width of the Gaussian filter used on the time series.
        The inverse width is :math:`C_\delta` times the spacing of the frequency grid.

    discrepency_factor: float
        Parameter :math:`C_d` used to determine the expected weighted error on the time series.
        The expected weighted error is :math:`C_d` times the total number of data points (which, in turm, is twice the
        number of time points).

    adjust_moments: bool
        Whether to fix the norm, mean and varaince of the local density of states to their exact values after performing
        the quantile filter.

    norm_relative_weight: float
        The relative weight of the norm data point (i.e. t=0) with respect to the rest of the time series.
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        loschmidt_echo_evaluator: LoschmidtEchoEvaluator,
        t_max: float,
        n_t: int,
        n_omega: int = 256,
        omega_max: float = None,
        delta_factor: float = 2,
        discrepency_factor: float = 1,
        norm_relative_weight: float = 10,
        adjust_moments: bool = True,
    ):
        self.model = model
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
        self.delta = self.d_omega * delta_factor
        self.norm_relative_weight = norm_relative_weight
        self.discrepency_factor = discrepency_factor
        self.adjust_moments = adjust_moments

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
        G, err = self.loschmidt_echo_evaluator.evaluate(psi_bitstring, self.t)
        G_weights = 1.0 / np.maximum(err.real, ERR_EPSILLON) + 1j / np.maximum(err.imag, ERR_EPSILLON)
        G = filter_timeseries(G, self.t, self.delta)
        G_weights = filter_timeseries(G_weights, self.t, self.delta)
        moments_weight = self.norm_relative_weight * (np.mean(G_weights.real) + np.mean(G_weights.imag)) / 2.0
        D, q = non_negative_least_squares_filtered(
            self.omega,
            len(G) * 2 * self.discrepency_factor,
            t=self.t,
            G_t=G,
            G_t_weights=G_weights,
            moments_weight=moments_weight,
        )

        # Adjust moments
        if self.adjust_moments:
            psi = bitstring_to_state(psi_bitstring)
            exact_mean = self.model.calc_energy_mean(psi)
            exact_width = math.sqrt(self.model.calc_energy_moment(psi, 2) - exact_mean**2 + self.delta**2)
            return adjust_moments(D, self.omega, self.d_omega, exact_mean, exact_width)
        else:
            return D, self.omega, self.d_omega

    def calculate(self, psi_bitstring: list) -> tuple[float, float, float]:
        """Inherited method. See :py:meth:`BoltzmannWeightCalculator.calculate`."""
        D, omega, d_omega = self.estimate_density(psi_bitstring)
        res = integrate_density(D, omega, d_omega, self.beta, self.delta)
        return res

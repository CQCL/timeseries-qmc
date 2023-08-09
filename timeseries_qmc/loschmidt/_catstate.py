from typing import Union
import math
import numpy as np
from pytket.backends.backend import Backend
from ..models import Model
from ..mitigation import SymmetryFilter
from ..utils import bitstring_to_int, bitstring_to_state
from ._evaluator import LoschmidtEchoEvaluator
from ._cache import FilesShotsCache, MemoryShotsCache


class Catstate(LoschmidtEchoEvaluator):
    r"""Evaluates Loschmidt echos using of a model Trotterized quantum circuits via the cat-state trick.

    The error estimates of this evaluator are the standard deviations of shot noise.
    Totterization error is not taken into account.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamiltonain.

        .. warning:: This implementation of the cat-state trick assumes that :math:`|0\dots 0\rangle` is an eigenstate
                     of the Hamiltonian. It is the responsibility of the user to ensure that the supplied model (even
                     after Trotterization) satifies this property. 

    backend: :class:`pytket.backends.backend.Backend`
        Backend used to run the quantum circuits.

    n_shots: int
        Number of shots used for each circuit.

    cache_directroy: str
        Directory used to cache compiled circuits, job handles and shots.
        When ``None`` is used, objects are cached in memory.

    trotter_order: int
        The order of Tortterization. Currently, only 1st and 2nd orders are supported.

    n_steps: int
        Number of Trotter steps. Either this parameter or **dt_trotter** needs to be specified.
        Both parameters cannot be specified simultaneously.

    dt_trotter: float
        Size of Trotter step. Either this parameter or **n_steps** needs to be specified.
        Both parameters cannot be specified simultaneously.

    use_energy_shift: bool
        Whether to shift the Hamiltonain by the energy of the state before running its circuits.
        When this happens, the Loschmidt echos will be shifted back.

    log_preperation: bool
        Whether to use a circuit of logarithmic depth for prepering the cat state. 
        Deafult value is true (recommended).

    symmetry_filter: :class:`.mitigation.SymmetryFilter`
        Filter for the shots of a noisy quantum circuit that violate the symmetries of the models.
        If ``None`` is used, the backend is assumed to be noiseless and shots are filtered.

        .. warning:: It is the responsibility of the user to ensure that the supplied **symmetry_filter** actually
                     describes the symmetries of the supplied **model**.
    """
    def __init__(
        self,
        model: Model,
        backend: Backend,
        n_shots: int,
        cache_directory: str = None,
        trotter_order: int = 1,
        n_steps: int = None,
        dt_trotter: float = None,
        use_energy_shift: bool = False,
        log_preperation: bool = True,
        symmetry_filter: SymmetryFilter = None,
    ):
        self.model = model
        if cache_directory:
            self.cache = FilesShotsCache(cache_directory, n_shots, backend)
        else:
            self.cache = MemoryShotsCache(n_shots, backend)

        self.trotter_order = trotter_order
        self.n_steps = n_steps

        self.dt_trotter = dt_trotter
        if dt_trotter is None and n_steps is None:
            raise ValueError("Either dt_trotter or n_steps should be specified")
        if dt_trotter is not None and n_steps is not None:
            raise ValueError("Only dt_trotter or n_steps can be specified")

        self.use_energy_shift = use_energy_shift
        self.log_preperation = log_preperation

        self.symmetry_filter = symmetry_filter

    def evaluate(self, psi_bistring: Union[list, np.ndarray], t_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inherited method. See :py:meth:`LoschmidtEchoEvaluator.evaluate`."""
        if not np.any(psi_bistring) or np.all(psi_bistring):
            # We have either state |0..0> or |1..1> which we know are eigenstates (otherwise cat circuit shouldn't be used)
            psi = bitstring_to_state(psi_bistring)
            E = self.model.calc_energy_mean(psi)

            G = np.zeros(t_points.size, dtype=np.cdouble)
            G_err = np.zeros(t_points.size, dtype=np.cdouble)
            for i in range(t_points.size):
                G[i], G_err[i] = np.exp(-1j * E * t_points[i]), 0.0 + 1j * 0.0
            return G, G_err

        if self.use_energy_shift:
            psi = bitstring_to_state(psi_bistring)
            energy_shift = self.model.calc_moment(psi, 1)
        else:
            energy_shift = 0

        for i in range(t_points.size):
            self.cache.prepare_shots(self._get_circuit(psi_bistring, t_points[i], False, energy_shift))
            self.cache.prepare_shots(self._get_circuit(psi_bistring, t_points[i], True, energy_shift))

        G = np.zeros(t_points.size, dtype=np.cdouble)
        G_err = np.zeros(t_points.size, dtype=np.cdouble)
        for i in range(t_points.size):
            G[i], G_err[i] = self._evaluate_timepoint(psi_bistring, t_points[i], energy_shift)
        return G, G_err

    def _get_circuit(self, psi_bistring, t, imaginary, energy_shift):
        if self.n_steps is not None:
            n_steps = self.n_steps
        else:
            n_steps = math.ceil(abs(t) / self.dt_trotter)
            n_steps = max(1, n_steps)

        circuit = self.model.make_cat_circuit(
            psi_bistring, t, n_steps, self.trotter_order, imaginary, energy_shift, self.log_preperation
        )
        if imaginary:
            circuit.name = "psi={},t={:.3f},imag".format(bitstring_to_int(psi_bistring), t)
        else:
            circuit.name = "psi={},t={:.3f},real".format(bitstring_to_int(psi_bistring), t)
        circuit.measure_all()
        return circuit

    def _evaluate_timepoint(self, psi_bistring, t, energy_shift):
        real_circuit = self._get_circuit(psi_bistring, t, False, energy_shift)
        imag_circuit = self._get_circuit(psi_bistring, t, True, energy_shift)
        real_shots = self.cache.get_shots(real_circuit)
        imag_shots = self.cache.get_shots(imag_circuit)
        if self.symmetry_filter:
            real_shots = self.symmetry_filter.filter_shots(real_shots, psi_bistring, self.log_preperation)
            imag_shots = self.symmetry_filter.filter_shots(imag_shots, psi_bistring, self.log_preperation)

        real_val, real_err = self._calc_val_err(real_shots, psi_bistring)
        imag_val, imag_err = self._calc_val_err(imag_shots, psi_bistring)

        # undo the energy shift i.e. multipy by exp(-1j*energy_shift*t)
        c = math.cos(-energy_shift * t)
        s = math.sin(-energy_shift * t)
        true_val_real = real_val * c - imag_val * s
        true_val_imag = real_val * s + imag_val * c

        true_err_real = math.sqrt(real_err**2 * c**2 + imag_err**2 * s**2)
        true_err_imag = math.sqrt(real_err**2 * s**2 + imag_err**2 * c**2)

        return true_val_real + 1j * true_val_imag, true_err_real + 1j * true_err_imag

    def _calc_val_err(self, shots, psi_bitstring, scale_factor=1):
        shots = list(shots)
        state_0 = np.zeros(self.model.n_qbits)
        state_pi = np.zeros(self.model.n_qbits)
        ctrl_qbit = list(psi_bitstring).index(1)
        state_pi[ctrl_qbit] = 1

        p_0 = sum([1 if (shot == state_0).all() else 0 for shot in shots]) / len(shots)
        p_pi = sum([1 if (shot == state_pi).all() else 0 for shot in shots]) / len(shots)

        val = p_0 - p_pi
        # Var[p_0-p_pi] = Var[p_0] + Var[p_pi] - 2* Cov[p_0, p_pi]
        # Var[p_0] = p_0*(1-p_0)/N
        # Var[p_pi] = p_pi*(1-p_pi)/N
        # Cov[p_0, p_pi] = -p_0*p_1/N (Covraince of multinomial distribution)
        err = math.sqrt((p_0 * (1 - p_0) + p_pi * (1 - p_pi) + 2 * p_0 * p_pi) / len(shots))

        return val / scale_factor, err / scale_factor

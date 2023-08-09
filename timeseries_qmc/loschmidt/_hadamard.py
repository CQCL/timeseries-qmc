from typing import Union
import math
import numpy as np
from pytket.backends.backend import Backend
from ..models import Model
from ..mitigation import FidelityEstimator
from ..utils import bitstring_to_circuit, bitstring_to_int, bitstring_to_state
from ._evaluator import LoschmidtEchoEvaluator
from ._cache import FilesShotsCache, MemoryShotsCache


class Hadamard(LoschmidtEchoEvaluator):
    """Evaluates Loschmidt echos using of a model Trotterized quantum circuits via the Hadamard test.

    The error estimates of this evaluator are the standard deviations of shot noise.
    Totterization error is not taken into account.

    Parameters
    ----------
    model: :class:`.models.Model`
        Model Hamiltonain.

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

    n_controls: int
        Number of control qubits used in the Hadamard test. It should be greater than zero.

    split: bool
        Whether to split 2nd-order Trotter cicuit or not. See :py:meth:`.models.Model.make_split_2nd_trotter_circuits` 
        for details. This paramter is ignored for 1st-order circuits.

    use_optimized_control: bool
        Whether to use manually-optimized controlled gates or rely on pytket QControlBox.
    
    fidelity_estimator: :class:`.mitigation.FidelityEstimator`
        Estimator of the total fidelity of a quantum circuit (aka q-factor).
        This can be used to rescale Loschmidt echo, when the Backend is a noisy one.
        If ``None`` is used, the backend is assumed to be noiseless.

        .. warning:: It is the responsibility of the user to ensure that the supplied **fidelity_estimator** actually
                     describes the fidelity of the supplied **backend**.
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
        n_controls: int = 1,
        split: bool = True,
        use_optimized_control: bool = True,
        fidelity_estimator: FidelityEstimator = None,
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
        if(n_controls<1):
            raise ValueError("n_controls should be greater than zero.")
        
        self.n_controls = n_controls
        self.split = split
        self.use_optimized_control = use_optimized_control

        self.fidelity_estimator = fidelity_estimator

    def evaluate(self, psi_bistring: Union[list, np.ndarray], t_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inherited method. See :py:meth:`LoschmidtEchoEvaluator.evaluate`."""
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
        psi_circuit = bitstring_to_circuit(psi_bistring)
        if self.n_steps is not None:
            n_steps = self.n_steps
        else:
            n_steps = math.ceil(abs(t) / self.dt_trotter)
            n_steps = max(1, n_steps)

        circuit = self.model.make_hadamard_circuit(
            psi_circuit,
            t,
            n_steps,
            self.trotter_order,
            self.split,
            self.use_optimized_control,
            self.n_controls,
            imaginary,
            energy_shift,
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
        if self.fidelity_estimator:
            compiled_real_circuit = self.cache.get_compiled_circuit(real_circuit)
            compiled_imag_circuit = self.cache.get_compiled_circuit(imag_circuit)
            real_scale = self.fidelity_estimator.estimate_fidelity(compiled_real_circuit)
            imag_scale = self.fidelity_estimator.estimate_fidelity(compiled_imag_circuit)
        else:
            real_scale = 1.0
            imag_scale = 1.0

        real_val, real_err = self._calc_val_err(real_shots, real_scale)
        imag_val, imag_err = self._calc_val_err(imag_shots, imag_scale)

        # undo the energy shift i.e. multipy by exp(-1j*energy_shift*t)
        c = math.cos(-energy_shift * t)
        s = math.sin(-energy_shift * t)
        true_val_real = real_val * c - imag_val * s
        true_val_imag = real_val * s + imag_val * c

        true_err_real = math.sqrt(real_err**2 * c**2 + imag_err**2 * s**2)
        true_err_imag = math.sqrt(real_err**2 * s**2 + imag_err**2 * c**2)

        return true_val_real + 1j * true_val_imag, true_err_real + 1j * true_err_imag

    def _calc_val_err(self, shots, scale_factor):
        shots_count = len(shots)
        ones_count = np.count_nonzero(shots[:, 0])
        # Add one shot of each outcome. This is a trick to avoid zero error estimate when all shots are the same
        shots_count += 2
        ones_count += 1
        p1 = ones_count / shots_count
        val = 1 - 2 * p1
        err = math.sqrt(4 * (p1 * (1 - p1)) / shots_count)

        return val / scale_factor, err / scale_factor

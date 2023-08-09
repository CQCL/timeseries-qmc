from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pytket
from pytket.circuit import Circuit
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
from ._circuits import make_cat_circuit, make_hadamard_circuit
from ..utils import *


class Model(ABC):
    """Abstract base class representing a Trotterized model Hamiltonian.
    It can be used to obtian its quantum circuits, Quspin operators and other properites.
    """

    @property
    @abstractmethod
    def n_qbits(self) -> int:
        """Number of qubits necessary for representing a state of this model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_terms(self) -> int:
        """Number of non-commutative terms in the Hamilatonian."""
        raise NotImplementedError()

    @abstractmethod
    def append_gates(self, qc: pytket.circuit.Circuit, term: int, t: float, n_controls: int) -> None:
        """Append gates that implements the time evolution according to specified term of the Hamiltonian.

        Parameters
        ----------
        qc: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit to append the gates to.
        term: int
            Index of the Hamiltonian term.
        t: float
            Time to evolve.
        n_controls: int
            The number of ancillary control qbits in the circuit. If greater than zero, a controlled verison of gates 
            are appended. The method is free to distribuit the control on these qbits to maximize parallel excuation.
            The control qbits are assumed to be the first qbits in the circuit.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_hamiltonian(self, term: int) -> hamiltonian:
        """Prepare a QuSpin operator representing the specified term of the Hamiltonian.

        Parameters
        ----------
        term: int
            Index of the Hamiltonain term.

        Returns
        -------
        h_term: :class:`quspin.operators.hamiltonian`
        """
        raise NotImplementedError()

    def get_full_hamiltonian(self) -> hamiltonian:
        """Prepare a QuSpin operator representing the full Hamiltonian.

        Returns
        -------
        h: :class:`quspin.operators.hamiltonian`
        """

        H = self.get_hamiltonian(0)
        for term in range(1, self.n_terms):
            H = H + self.get_hamiltonian(term)
        return H

    def make_1st_trotter_circuit(self, n_steps: int, dt: float, n_controls: int = 0) -> Circuit:
        r"""Return a pytket circuit that implements the 1st-order Trotterized time evolution of the model.

        Parameters
        ----------
        n_steps: int
            Number of Trotter steps :math:`n`.

        dt: float
            Trotter step size :math:`\delta t`.

        n_controls: int
            The number of ancillary control qbits in the circuit. If greater than zero, return the circuit with an 
            extra qbits for controlling the gates. The method is free to distribuit the control on these qbits to 
            maximize parallel excuation.The control qbits are assumed to be the first qbits in the circuit.


        Returns
        -------
        circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit implementing first-order Trotterization  :math:`\left[e^{-i D \delta t} \dots e^{-i B \delta t} 
            e^{-i A \delta t}\right]^n`
            where :math:`A, B, \dots, D` are non-commuting terms  of the Hamiltonain.
        """
        size = self.n_qbits + n_controls
        name = "Controlled Trotter" if (n_controls > 0) else "Trotter"

        qc = Circuit(size, name=name)

        for _ in range(n_steps):
            for term in range(0, self.n_terms):
                self.append_gates(qc, term, dt, n_controls)

        return qc

    def make_2nd_trotter_circuit(self, n_steps: int, dt: float, n_controls: int = 0) -> Circuit:
        r"""Return a pytket circuit that implements the 2nd-order Trotterized time evolution of the model.

        Parameters
        ----------
        n_steps: int
            Number of Trotter steps :math:`n`.

        dt: float
            Trotter step size :math:`\delta t`.

        n_controls: int
            The number of ancillary control qbits in the circuit. If greater than zero, return the circuit with an 
            extra qbits for controlling the gates. The method is free to distribuit the control on these qbits to 
            maximize parallel excuation. The control qbits are assumed to be the first qbits in the circuit.


        Returns
        -------
        circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit implementing second-order Trotterization  :math:`\left[e^{-i A \delta t/2}e^{-i B \delta t/2}\dots 
            e^{-i D \delta t}\dots e^{-i B \delta t/2}e^{-i A \delta t/2}\right]^n`
            where :math:`A, B, \dots, D` are non-commuting terms  of the Hamiltonain.
        """

        size = self.n_qbits + n_controls
        name = "Controlled Trotter" if (n_controls > 0) else "Trotter"
        n_terms = self.n_terms

        qc = Circuit(size, name=name)
        if n_steps == 0:
            return qc

        # A/2...C/2
        for term in range(0, n_terms - 1):
            self.append_gates(qc, term, dt / 2.0, n_controls)

        # (D C/2 B/2 A B/2 C/2)^(n-1)
        for _ in range(n_steps - 1):
            self.append_gates(qc, n_terms - 1, dt, n_controls)
            for term in reversed(range(1, n_terms - 1)):
                self.append_gates(qc, term, dt / 2.0, n_controls)
            self.append_gates(qc, 0, dt, n_controls)
            for term in range(1, n_terms - 1):
                self.append_gates(qc, term, dt / 2.0, n_controls)

        # D C/2 B/2 A/2
        self.append_gates(qc, n_terms - 1, dt, n_controls)
        for term in reversed(range(0, n_terms - 1)):
            self.append_gates(qc, term, dt / 2.0, n_controls)

        return qc

    def make_split_2nd_trotter_circuits(self, n_steps: int, dt: float, n_controls: int = 0) -> tuple[Circuit, Circuit]:
        r"""Return pytket circuits that implements the splitted 2nd-order Trotterized time evolution of the model.

        **Example** A second-order Trotterization :math:`[e^{-i A \delta t/2} e^{-i B \delta t} e^{-i A \delta t/2}]^n`
        can written as :math:`e^{i A \delta t/2} [e^{-i A \delta t} e^{-i B  \delta t}]^n e^{-i A \delta t/2}`.
        This splitting is useful because it allows performing the Hadamard test
        by controlling only the circuit of :math:`[e^{-i A \delta t} e^{-i B  \delta t}]^n` while considering the 
        circuit of :math:`e^{-i A \delta t/2}` as part of state preperation.

        Parameters
        ----------
        n_steps: int
            Number of Trotter steps :math:`n`.

        dt: float
            Trotter step size :math:`\delta t`.

        n_controls: int
            The number of ancillary control qbits in the circuit. If greater than zero, return the circuit with an 
            extra qbits for controlling the gates. The method is free to distribuit the control on these qbits to 
            maximize parallel excuation. The control qbits are assumed to be the first qbits in the circuit.


        Returns
        -------
        main_circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit implementing the Trotterization :math:`\left[e^{-i C \delta t/2} \dots e^{-i B \delta t/2} 
            e^{-i A \delta t} e^{-i B \delta t/2} \dots e^{-i C \delta t/2} e^{-i D \delta} \right]^n`
            where :math:`A, B, \dots, C, D` are non-commuting terms  of the Hamiltonain.

        residue_circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit implementing the Trotterization :math:`\left[e^{-i C \delta t/2}  \dots e^{-i B \delta t/2} 
            e^{-i A \delta t/2} \right]`.
        """
        size = self.n_qbits + n_controls
        name = "Controlled Trotter" if (n_controls > 0) else "Trotter"
        n_terms = self.n_terms

        qc1 = Circuit(size, name=name + " - Main")
        qc2 = Circuit(self.n_qbits, name="Trotter - Residue")
        if n_steps == 0:
            return qc1, qc2

        # (D C/2 B/2 A B/2 C/2)^n
        for _ in range(n_steps):
            self.append_gates(qc1, n_terms - 1, dt, n_controls)
            for term in reversed(range(1, n_terms - 1)):
                self.append_gates(qc1, term, dt / 2.0, n_controls)
            self.append_gates(qc1, 0, dt, n_controls)
            for term in range(1, n_terms - 1):
                self.append_gates(qc1, term, dt / 2.0, n_controls)

        # A/2...C/2
        for term in range(0, n_terms - 1):
            self.append_gates(qc2, term, dt / 2.0, 0)

        return qc1, qc2

    def make_hadamard_circuit(
        self,
        psi_circuit: Circuit,
        t: float,
        n_steps: int,
        trotter_order: int = 2,
        split: bool = True,
        use_optimized_control: bool = True,
        n_controls: int = 1,
        imaginary: bool = False,
        energy_shift=0,
    ) -> Circuit:
        r"""Return a pytket circuit that implements the Hadamard test using specified Trotterization of H.

        Parameters
        ----------
        psi_circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit for transforming the :math:`|0\dots0\rangle` state into :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        n_steps: int
            Number of Trotter steps :math:`n`.

        trotter_order : int
            The order of Tortterization:math:`p`. Currently, only 1st and 2nd orders are supported.

        split: bool
            Whether to split 2nd-order Trotter cicuit or not. See :py:meth:`make_split_2nd_trotter_circuits` for details.
            This paramter is ignored for 1st-order circuits.

        use_optimized_control: bool
            Whether to use manually-optimized controlled gates or rely on pytket QControlBox.

        n_controls: int
            The number of ancillary control qbits in the circuit. Using more than one control qbit has the potential of 
            speed-up by parallelizing excution of controlled gates.
            If 'use_optimized_control' is False, this value is ignored and only one control qbit is used.
            The control qbits are the first qbits in the circuit.

        imaginary: bool
            If 'True', return the circuit measuring the imaginary part of Loschmidt echo. Otherwise, return the circuit 
            for the real part.

        energy_shift: float
            Value to be used for shifting the zero energy of the Hamiltonian :math:`E`. Default value is zero.

        Returns
        -------
        circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Hadmard circuit for measuring the real or imaginary part of 
            :math:`\langle\psi| e^{-i t E} U_p^n(t) |\psi\rangle`
            where :math:`U_1(t) := e^{-i D t/n}\dots e^{-i B t/n}e^{-i A t/n}` is first-order Trotterization,
            :math:`U_2(t) := e^{-i A t/(2n)}e^{-i B t/(2n)}\dots e^{-i D t/n}\dots e^{-i B t/(2n)}e^{-i A t/(2n)}` is 
            second-order Trotterization and :math:`A, B, \dots, D` are non-commuting terms  of the Hamiltonain.
        """

        dt = t / n_steps
        n_manual_controls = n_controls if use_optimized_control else 0
        if trotter_order == 1:
            trotter_circuit = self.make_1st_trotter_circuit(n_steps, dt, n_manual_controls)
        elif trotter_order == 2:
            if split:
                trotter_circuit, residue = self.make_split_2nd_trotter_circuits(n_steps, dt, n_manual_controls)
                psi_circuit = psi_circuit.copy()
                psi_circuit.append(residue)
            else:
                trotter_circuit = self.make_2nd_trotter_circuit(n_steps, dt, n_manual_controls)
        else:
            raise ValueError("Unsupported Trotterization order")
        return make_hadamard_circuit(psi_circuit, trotter_circuit, n_manual_controls, imaginary, energy_shift * t)

    def make_amplitude_circuit(self, psi_circuit: Circuit, t: float, n_steps: int, trotter_order: int = 2) -> Circuit:
        r"""Return a pytket circuit that calculates the squared amplitude of Loschmidt echo for the specified state.

        Parameters
        ----------
        psi_circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Circuit for transforming the :math:`|0\dots0\rangle` state into :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        n_steps: int
            Number of Trotter steps :math:`n`.

        trotter_order : int
            The order of Tortterization :math:`p`. Currently, only 1st and 2nd orders are supported.

        Returns
        -------
        circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Amplitude circuit for measuring :math:`|\langle\psi| e^{-i t E} U_p^n(t) |\psi\rangle|^2`
            where :math:`U_1(t) := e^{-i D t/n}\dots e^{-i B t/n}e^{-i A t/n}` is first-order Trotterization,
            :math:`U_2(t) := e^{-i A t/(2n)}e^{-i B t/(2n)}\dots e^{-i D t/n}\dots e^{-i B t/(2n)}e^{-i A t/(2n)}` 
            is second-order Trotterization and :math:`A, B, \dots, D` are non-commuting terms  of the Hamiltonain.
        """
        dt = t / n_steps
        n_qbits = self.n_qbits
        qc = Circuit(n_qbits)
        psi = list(range(n_qbits))
        qc.add_circuit(psi_circuit, psi)
        if trotter_order == 1:
            qc.add_circuit(self.make_1st_trotter_circuit(n_steps, dt, 0), psi)
        elif trotter_order == 2:
            qc.add_circuit(self.make_2nd_trotter_circuit(n_steps, dt, 0), psi)
        else:
            raise ValueError("Unsupported Trotterization order")
        qc.add_circuit(psi_circuit.dagger(), psi)
        return qc

    def make_cat_circuit(
        self,
        psi_bitstring: Union[list, np.ndarray],
        t: float,
        n_steps: int,
        trotter_order: int = 2,
        imaginary: bool = False,
        energy_shift=0,
        log_preperation=True,
    ) -> Circuit:
        r"""Return a pytket circuit that implements the catstate trick for calculating Loschmidt echo for the specified
          state.

        .. warning:: This method assumes that the state :math:`|0\dots0\rangle` is an eigenstate of the model. 
                     Otherwise, the results are unpredictable.

        Parameters
        ----------
        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing some product state :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        n_steps: int
            Number of Trotter steps :math:`n`.

        trotter_order : int
            The order of Tortterization :math:`p`. Currently, only 1st and 2nd orders are supported.

        imaginary: bool
            If 'True', return the circuit measuring the imaginary part of Loschmidt echo. 
            Otherwise, return the circuit for the real part.

        energy_shift: float
            Value to be used for shifting the zero energy of the Hamiltonian :math:`E`. Default value is zero.

        log_preperation: bool
            Whether to use a circuit of logarithmic depth for prepering the cat state. 
            Deafult value is true (recommended).

        Returns
        -------
        circuit: :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>`
            Catstate circuit for measuring the real or imaginary part of 
            :math:`\langle\psi| e^{-i t E} U_p^n(t) |\psi\rangle`
            where :math:`U_1(t) := e^{-i D t/n}\dots e^{-i B t/n}e^{-i A t/n}` is first-order Trotterization,
            :math:`U_2(t) := e^{-i A t/(2n)}e^{-i B t/(2n)}\dots e^{-i D t/n}\dots e^{-i B t/(2n)}e^{-i A t/(2n)}` is 
            second-order Trotterization and :math:`A, B, \dots, D` are non-commuting terms  of the Hamiltonain.
        """

        dt = t / n_steps
        if trotter_order == 1:
            trotter_circuit = self.make_1st_trotter_circuit(n_steps, dt)
        elif trotter_order == 2:
            trotter_circuit = self.make_2nd_trotter_circuit(n_steps, dt)
        else:
            raise ValueError("Unsupported Trotterization order")
        psi_0 = np.zeros(2**self.n_qbits)
        psi_0[0] = 1
        E0 = self.calc_energy_mean(psi_0)
        if imaginary:
            theta = -np.pi / 2
        else:
            theta = 0.0
        # Cat state trick assumes that energy of |0..0> is zero, so we need to additioanlly shift the phase accoringly
        return make_cat_circuit(
            trotter_circuit,
            psi_bitstring,
            theta,
            (-E0 + energy_shift) * t,
            log_preperation,
        )

    def evolve_state_exact(self, psi: np.ndarray, t: float) -> np.ndarray:
        r"""Apply the exact time evolution of the model on a state and return the evolved state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        Returns
        -------
        psi_t: :py:class:`numpy.ndarray`
            Evolved state vector :math:`e^{-iHt} |\psi\rangle`
        """
        evolve = expm_multiply_parallel(self.get_full_hamiltonian().tocsr(), -1j * t)
        psi = evolve.dot(psi)
        return psi

    def evolve_state_1st_trotter(self, psi: np.ndarray, t: float, n_steps: int) -> np.ndarray:
        r"""Apply the 1st-order Trotterized time evolution of the model on a state and return the evolved state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        n_steps: int
            Number of Trotter steps :math:`n`.

        Returns
        -------
        psi_t: :py:class:`numpy.ndarray`
            Evolved state vector :math:`U_1^n(t)|\psi\rangle`,
            where :math:`U_1(t) := e^{-i A t/n}e^{-i B t/n}\dots e^{-i D t/n}` is first-order Trotterization of
            Hamiltonain with non-commuting terms :math:`A, B, \dots, D`.
        """
        if n_steps == 0:
            return np.array(psi)

        dt = t / n_steps
        evolve_dt = []

        for term in range(self.n_terms):
            mat = self.get_hamiltonian(term).tocsr()
            evolve_dt.append(expm_multiply_parallel(mat, -1j * dt))

        for _ in range(n_steps):
            for term in range(0, self.n_terms):
                psi = evolve_dt[term].dot(psi)

        return psi

    def evolve_state_2nd_trotter(self, psi: np.ndarray, t: float, n_steps: int) -> np.ndarray:
        r"""Apply the 2nd-order Trotterized time evolution of the model on a state and return the evolved state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        t: float
            Time to evolve :math:`t`.

        n_steps: int
            Number of Trotter steps :math:`n`.

        Returns
        -------
        psi_t: :py:class:`numpy.ndarray`
            Evolved state vector :math:`U_2^n(t)|\psi\rangle`,
            where :math:`U_2(t) := e^{-i A t/(2n)}e^{-i B t/(2n)}\dots e^{-i D t/n}\dots e^{-i B t/(2n)}e^{-i A t/(2n)}`
            is second-order Trotterization of Hamiltonain with non-commuting terms :math:`A, B, \dots, D`.
        """
        if n_steps == 0:
            return np.array(psi)

        dt = t / n_steps
        evolve_dt = []
        evolve_half_dt = []
        n_terms = self.n_terms

        for term in range(n_terms):
            mat = self.get_hamiltonian(term).tocsr()
            if term == 0 or term == n_terms - 1:
                evolve_dt.append(expm_multiply_parallel(mat, -1j * dt))
            else:
                # We only need to evolve the first and last term by dt
                evolve_dt.append(None)
            if term != n_terms - 1:
                evolve_half_dt.append(expm_multiply_parallel(mat, -1j * dt / 2.0))
            else:
                # We don't need to evlove the last term by dt/2
                evolve_half_dt.append(None)

        # A/2...C/2
        for term in range(0, n_terms - 1):
            psi = evolve_half_dt[term].dot(psi)

        # (D C/2 B/2 A B/2 C/2)^(n-1)
        for _ in range(n_steps - 1):
            psi = evolve_dt[n_terms - 1].dot(psi)
            for term in reversed(range(1, n_terms - 1)):
                psi = evolve_half_dt[term].dot(psi)
            psi = evolve_dt[0].dot(psi)
            for term in range(1, n_terms - 1):
                psi = evolve_half_dt[term].dot(psi)

        # D C/2 B/2 A/2
        psi = evolve_dt[n_terms - 1].dot(psi)
        for term in reversed(range(0, n_terms - 1)):
            psi = evolve_half_dt[term].dot(psi)

        return psi

    def calc_boltzmann_weight(self, psi: np.ndarray, beta: float) -> tuple[float, float, float]:
        r"""Calculate the Boltzmann weight of a state and its local energy moments at the specified invserse 
        temperature.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        beta: float
            Inverse temperature :math:`\beta`.

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
        ham = self.get_full_hamiltonian().tocsr()
        exp_mbH_half = expm_multiply_parallel(ham, -beta / 2.0)
        psi_half_evolved = exp_mbH_half.dot(psi)
        w = psi_half_evolved.conj().dot(psi_half_evolved)
        psi_half_evolved_ham = ham.dot(psi_half_evolved)
        e = psi_half_evolved.conj().dot(psi_half_evolved_ham) / w
        e_sq = psi_half_evolved_ham.conj().dot(psi_half_evolved_ham) / w
        return w, e, e_sq

    def calc_energy_moment(self, psi: np.ndarray, n: int) -> float:
        r"""Calculate the expectation value of the n-th power of the Hamiltonian for the specified state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        n: int
            power of the Hamiltonian :math:`n`.

        Returns
        -------
        result: float
            Expectation value :math:`\langle\psi| H^n |\psi \rangle`.

        """
        ham = self.get_full_hamiltonian().tocsr()
        new_psi = psi
        for i in range(n):
            new_psi = ham.dot(new_psi)
        return psi.conj().dot(new_psi)

    def calc_energy_mean(self, psi: np.ndarray) -> float:
        r"""Calculate the energy expectation value for the specified state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        Returns
        -------
        result: float
            Energy expectation value :math:`\langle\psi| H |\psi \rangle`.
        """
        return self.calc_energy_moment(psi, 1)

    def calc_energy_variance(self, psi: np.ndarray) -> float:
        r"""Calculate the energy variance for the specified state.

        Parameters
        ----------
        psi: :py:class:`numpy.ndarray`
            State vector :math:`|\psi\rangle`.

        Returns
        -------
        result: float
            Energy variance :math:`\langle\psi| H^2 |\psi \rangle - \langle\psi| H |\psi \rangle^2`.
        """
        energy_mean = self.calc_energy_mean(psi)
        return self.calc_energy_moment(psi, 2) - energy_mean**2
    
    def calc_thermal_observables(self, beta:float) -> tuple[float, float, float]:
        r"""Calculates the magnetizaiton squared and the first two moments of energy at a finite temperature.

        .. warning:: This performes an exact diagonalization of the full hamiltonain, so it should be used only for
                    small system sizes.

        Parameters
        ----------
        beta : float
            Inverse temperature :math:`\beta`.

        Returns
        -------
        M2: float
            Magnetization squared: :math:`\operatorname{Tr} \left[e^{-\beta H} (\sum_i Z_i)^2\right]/
            \operatorname{Tr} \left[e^{-\beta H}\right]`.
        E: flaot
            Energy: :math:`\operatorname{Tr} \left[e^{-\beta H} H\right]/
            \operatorname{Tr} \left[e^{-\beta H}\right]`.
        E2: float
            Energy squared: :math:`\operatorname{Tr} \left[e^{-\beta H} H^2\right]/
            \operatorname{Tr} \left[e^{-\beta H}\right]`.
        """
        L = self.n_qbits
        ##############################################################################
        # Diagonalize Hamiltonian
        ##############################################################################
        H = self.get_full_hamiltonian()
        E,V = H.eigh()
        ##############################################################################
        # Calculate Magnetization^2 of Eigenvectors
        ##############################################################################
        M_list =  [[1.0/L,i] for i in range(L)]
        M_op = hamiltonian([["z", M_list]],[], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False)
        M2_op = M_op**2
        M2 = M2_op.matrix_ele(V,V,diagonal=True)
        ##############################################################################
        # Calculate Finite Temperature Observales
        ##############################################################################
        mean_M2 = np.sum(M2*np.exp(-beta*E))/np.sum(np.exp(-beta*E))
        mean_E = np.real(np.sum(E * np.exp(-beta*E)) / np.sum(np.exp(-beta*E)))
        mean_E2 = np.sum(E**2*np.exp(-beta*E))/np.sum(np.exp(-beta*E))
        # Cv=(mean_E2-mean_E**2)*beta**2
        return mean_M2, mean_E, mean_E2
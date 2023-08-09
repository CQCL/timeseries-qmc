from abc import ABC, abstractmethod
from typing import Union
import random
import numpy as np
from ..utils import int_to_bitstring


class SymmetryFilter(ABC):
    """Abstract base Class for filtering shots of the catlike-state circuit that violate symmetry.

    Parameters
    ----------

    L: int
        Number of qubits.
    """

    def __init__(self, L: int):
        self.L = L

    def estimate_valid_ratio(
        self, psi_bitstring: Union[list, np.ndarray], samples_count: int = 100000, log: bool = True
    ) -> bool:
        """
        Estimate the ratio of random product states that violate the symmetry of the catlike-state circuit.

        Parameters
        ----------
        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of ones and zeros representation of the psi state which specifies the structure of the
            catlike-state circuit. Default value of ``None`` corresponds to antiferromagnetic state.

        samples_count: int
            Number of generated random product states.

        log: boolean
            Whether to assume linear- or logarithmic-depth catlike-state circuit.
        """
        if psi_bitstring == None:
            if self.L / 2 % 2 == 0:
                psi_bitstring = [0, 1] * int(self.L / 4) + [1, 0] * int(self.L / 4)
            else:
                psi_bitstring = [0, 1] * int(self.L / 4) + [0] + [1, 0] * int(self.L / 4) + [1]

        valid_count = 0
        for _ in range(samples_count):
            shot_bit_string = int_to_bitstring(random.randrange(2**self.L), self.L)
            if self.is_valid_shot(shot_bit_string, psi_bitstring, log):
                valid_count += 1

        return valid_count / samples_count

    def filter_shots(self, shots: np.ndarray, psi_bitstring: Union[list, np.ndarray], log: bool = True) -> np.ndarray:
        """
        Remove the shots of the catlike-state circuit that violate the symmetry.

        Parameters
        ----------
        shots: :py:class:`numpy.ndarray`
            2D array of ones and zeros representing the shots measured from the catelike-state circuit.

        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of ones and zeros representation of the psi state which specifies the structure of the
            catlike-state circuit.

        log: boolean
            Whether to assume linear- or logarithmic-depth catlike-state circuit.

        Returns
        -------
        filtered_shots: :py:class:`numpy.ndarray`
            2D array of integers representing the filtered shots that preserve the symmetry.
        """
        new_shots = []
        for shot in shots:
            if self.is_valid_shot(shot, psi_bitstring, log):
                new_shots.append(shot)
        return np.stack(new_shots, axis=0)

    def is_valid_shot(self, shot: Union[list, np.ndarray], psi_bitstring: Union[list, np.ndarray], log: bool) -> bool:
        """
        Checks whether the measured product state is a possible output of the catlike-state circuit

        Parameters
        ----------
        shot: List of int
            List/array of ones and zeros representing the shot measured from the catelike-state circuit.

        psi_bitstring: list or :py:class:`numpy.ndarray`
            List/array of ones and zeros representation of the psi state which specifies the structure of the
            catlike-state circuit.

        log: boolean
            Whether to assume linear- or logarithmic-depth catlike-state circuit.
        """
        assert len(psi_bitstring) == self.L
        assert len(shot) == self.L

        in_cat_state = self._get_cat_state_components([0] * self.L, psi_bitstring, log)
        out_cat_state = self._get_cat_state_components(shot, psi_bitstring, log)

        return self._is_symmetry_compatible(in_cat_state, out_cat_state)

    def _get_cat_state_components(self, input, psi, log):
        """
        Returns the components (list of bit strings) of the ouput of the catlike-state circuit
        when initialized with the specified input.

        Parameters
        ----------

        input: List of int
            Bit string representation of the input state.

        psi: List of int
            Bit string representation of the psi state which specifies the structure of the catlike-state circuit.

        log: boolean
            Whether to assume linear- or logarithmic-depth catlike-state circuit.
        """
        assert len(input) == self.L
        assert len(psi) == self.L

        if log:
            to_flip = []
            for i, s in enumerate(psi):
                if s == 1:
                    to_flip.append(i)
            if len(to_flip) < 1:
                # Bits of value 1 not found => psi is the vacuum
                return [input.copy()]

            first_control = to_flip[0]
            flipped = [first_control]
            to_flip.pop(0)
            control = {}  # Map each to-flip qubit to its controlling qubit
            while to_flip != []:
                nf = min(len(flipped), len(to_flip))
                for jq in range(nf):
                    control[to_flip[jq]] = flipped[jq]
                for jq in range(nf):
                    flipped.append(to_flip[jq])
                for jq in range(nf):
                    to_flip.pop(0)

            part0 = input.copy()
            part1 = input.copy()

            part0[first_control] = 0
            part1[first_control] = 1
            for i in range(first_control + 1, self.L):
                part0[i] = input[i] ^ (psi[i] and part0[control[i]])
                part1[i] = input[i] ^ (psi[i] and part1[control[i]])

            return [part0, part1]
        else:
            try:
                control_qbit = psi.index(1)
            except ValueError:
                # Bits of value 1 not found => psi is the vacuum
                return [input.copy()]

            part0 = input.copy()
            part0[control_qbit] = 0

            part1 = [input[i] ^ psi[i] for i in range(self.L)]
            part1[control_qbit] = 1

            return [part0, part1]

    def _is_symmetry_compatible(self, state1_components, state2_comonents):
        """
        Checks if two general states have compatible symmetries.

        Parameters
        ----------

        state1, state2: List of list of int
            List of all the product states (bit strings) that constitutes the state.
        """
        # Find at least a pair of bit strings that share the same symmetry
        for bit_string1 in state1_components:
            for bit_string2 in state2_comonents:
                if self.has_same_symmetry(bit_string1, bit_string2):
                    return True
        return False

    @abstractmethod
    def has_same_symmetry(self, bit_string1: Union[list, np.ndarray], bit_string2: Union[list, np.ndarray]) -> bool:
        """
        Check if two product states have the same symmetry

        Parameters
        ----------
        bit_string1: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing some product state.

        bit_string2: list or :py:class:`numpy.ndarray`
            List/array of zeros and ones representing anpther product state.
        """
        raise NotImplementedError()


class TotalSzFilter(SymmetryFilter):
    """Class for filtering shots of the catlike-state circuit that do not conserve the total Sz quantum number.

    Parameters
    ----------

    L: int
        Number of qubits.
    """

    def __init__(self, L: int):
        super().__init__(L)

    def has_same_symmetry(self, bit_string1: Union[list, np.ndarray], bit_string2: Union[list, np.ndarray]) -> bool:
        """Inherited method. See :py:meth:`SymmetryFilter.has_same_symmetry`."""
        return sum(bit_string1) == sum(bit_string2)


class ParticleNumberFilter(SymmetryFilter):
    """Class for filtering shots of the catlike-state circuit that do not conserve the total numbers of spin-up and spin-down electrons.

    .. Note:: This filter assumes a Jordan-Wigner transformation where the first half of qubits econdes spin-up
              electrons and the second half encodes spin-down electrons.

    Parameters
    ----------

    L: int
        Number of qubits.
    """

    def __init__(self, L: int):
        assert L % 2 == 0
        super().__init__(L)

    def has_same_symmetry(self, bit_string1: Union[list, np.ndarray], bit_string2: Union[list, np.ndarray]) -> bool:
        """Inherited method. See :py:meth:`SymmetryFilter.has_same_symmetry`."""
        N_sites = self.L // 2
        N_up1 = sum(bit_string1[:N_sites])
        N_up2 = sum(bit_string2[:N_sites])
        N_down1 = sum(bit_string1[N_sites:])
        N_down2 = sum(bit_string2[N_sites:])
        return N_up1 == N_up2 and N_down1 == N_down2

import numpy as np
import tenpy
import pytket
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from ..models._model import Model
from ..models._circuits import append_CRx, append_CZZPhase
from ..utils import get_coupling_list


class Ising(Model):
    """Class for obtaining quantum circuit Trotterized time evolution of a Transferse-field Ising model.

    Parameters
    ----------
    L: int
        Number of spins
    zz_list: list
        List of lists specifying the ZZ interaction part of the hamiltonian.
        The first element of each list is a float representing the coupling strength.
        The other two elements are integers representing the interacting sites.
    x_list: list
        List of lists specifying the X part of the hamiltonian.
        The first element of each list is a float representing the strength of magnatic field.
        The other element is an integer representing the site.
    which_first : str, optional
        Specifies whether the ZZ part or the X part should be applied first in Trotterization.
        Value can be either 'zz' or 'x'. Default value is 'zz'.
    """

    def __init__(self, L: int, zz_list: list[list], x_list: list[list], which_first: str = "zz"):
        self.L = L
        # Copy lists with opposite sign for a direct use in quspin
        self.zz_list = []
        for Jzz, site1, site2 in zz_list:
            self.zz_list.append([-Jzz, site1, site2])
        self.x_list = []
        for hx, site in x_list:
            self.x_list.append([-hx, site])
        self.basis = spin_basis_general(self.L)
        self.which_first = which_first

    @classmethod
    def from_coupling_list(cls, L: int, coupling_list: list[list], Jzz: float, hx: float, which_first: str = "zz"):
        """Shortcut method for creating an instance from a coupling list.

        Parameters
        ----------
        L: int
            Number of spins.
        coupling_list: list
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jzz : float
            Coupling strength in Z direction.
        hx : float
            Strength of magnatic field in X direction.
        which_first : str, optional
            Specifies whether the ZZ part or the X part should be applied first in Trotterization.
            Value can be either 'zz' or 'x'. Default value is 'zz'.

        Returns
        -------
        model: :class:`Ising`
            Ising model with the specified bonds and parameters.
        """
        zz_list = []
        x_list = []
        for i in range(len(coupling_list)):
            zz_list.append([Jzz, coupling_list[i][0], coupling_list[i][1]])
        for i in range(L):
            x_list.append([hx, i])

        return cls(L, zz_list, x_list, which_first)

    @classmethod
    def from_lattice(cls, lattice: tenpy.models.lattice.Lattice, Jzz: float, hx: float, which_first: str = "zz"):
        """Shortcut method for creating an instance from a lattice.

        Parameters
        ----------
        lattice: :class:`tenpy.models.lattice.Lattice`
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jzz : float
            Coupling strength in Z direction.
        hx : float
            Strength of magnatic field in X direction.
        which_first : str, optional
            Specifies whether the ZZ part or the X part should be applied first in Trotterization.
            Value can be either 'zz' or 'x'. Default value is 'zz'.

        Returns
        -------
        model: :class:`Ising`
            Ising model with the specified bonds and parameters.
        """
        coupling_list = get_coupling_list(lattice)
        L = len(lattice.order)
        return cls.from_coupling_list(L, coupling_list, Jzz, hx, which_first)

    @property
    def n_qbits(self) -> int:
        """Inherited property. see :py:attr:`Model.n_qbits`."""
        return self.L

    @property
    def n_terms(self) -> int:
        """Inherited property. see :py:attr:`Model.n_terms`."""
        return 2

    def _append_X_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for hx, qbit in self.x_list:
                append_CRx(qc, 2 / np.pi * t * hx, cnt_qbit, qbit + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for hx, qbit in self.x_list:
                qc.Rx(2 / np.pi * t * hx, qbit)

    def _append_ZZ_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for _Jzz, qbit1, qbit2 in self.zz_list:
                append_CZZPhase(qc, 2 / np.pi * t * _Jzz, cnt_qbit, qbit1 + n_controls, qbit2 + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for _Jzz, qbit1, qbit2 in self.zz_list:
                qc.ZZPhase(2 / np.pi * t * _Jzz, qbit1, qbit2)

    def append_gates(self, qc: pytket.circuit.Circuit, term: int, t: float, n_controls: int) -> None:
        """Inherited method. See :py:meth:`Model.append_gates`."""
        if term >= 2:
            raise ValueError("Ising has only two terms!")
        if (term == 0 and self.which_first == "zz") or (term == 1 and self.which_first == "x"):
            self._append_ZZ_gates(qc, t, n_controls)
        elif (term == 0 and self.which_first == "x") or (term == 1 and self.which_first == "zz"):
            self._append_X_gates(qc, t, n_controls)

    def _get_X_hamiltonian(self):
        return hamiltonian(
            [["x", self.x_list]], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False
        )

    def _get_ZZ_hamiltonian(self):
        return hamiltonian(
            [["zz", self.zz_list]], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False
        )

    def get_hamiltonian(self, term: int) -> hamiltonian:
        """Inherited method. See :py:meth:`Model.get_hamiltonian`."""
        if term >= 2:
            raise ValueError("Ising has only two terms!")
        if (term == 0 and self.which_first == "zz") or (term == 1 and self.which_first == "x"):
            return self._get_ZZ_hamiltonian()
        elif (term == 0 and self.which_first == "x") or (term == 1 and self.which_first == "zz"):
            return self._get_X_hamiltonian()

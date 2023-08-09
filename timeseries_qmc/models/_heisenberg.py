import numpy as np
import tenpy
import pytket
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from ..models._model import Model
from ..models._circuits import (
    append_CRz,
    append_CXXPhase,
    append_CYYPhase,
    append_CZZPhase,
    append_CXXPhaseYYPhaseZZPhase,
    append_CXXPhaseYYPhase,
)
from ..utils import get_coupling_list, get_parallel_couplings


class HeisenbergBonded(Model):
    """Class for handling Trotterized time evolution of a general Heisenberg model.
    Tortterization is done by seperating terms into mutally-exclusive sets of bonds.
    For an alternative Trotterization, see :class:`HeisenbergXYZ`.

    .. note:: The Trotterization of this model conserves the total Sz symmetry when Jx=Jy.

    Parameters
    ----------
    L: int
        Number of spins.
    bonds_lists: list
        List of lists of lists specifying the interaction part of the hamiltonian.
        Each sublist represents a set of mutally-exclusive set of bonds.
        The first element of each sub-sublist is a float representing the coupling strength Jx.
        The second element is a float representing the coupling strength Jy.
        The third element is a float representing the coupling strength Jz.
        The last two elements are integers representing the interacting sites.
    z_list: list
        List of lists specifying the Z part of the hamiltonian.
        The first element of each list is a float representing the strength of magnatic field.
        The other element is an integer representing the site.
    z_first : bool, optional
        Specifies whether the Z parts should be applied first in Trotterization.
        Default value is 'False'.
    """

    def __init__(self, L: int, bonds_lists: list[list[list]], z_list: list[list], z_first: bool = False):
        self.L = L
        self.bonds_lists = []
        # Copy bonds list with opposite sign for direct use in quspin
        for bond_list in bonds_lists:
            new_bond_list = []
            for Jxx, Jyy, Jzz, site1, site2 in bond_list:
                new_bond_list.append([-Jxx, -Jyy, -Jzz, site1, site2])
            self.bonds_lists.append(new_bond_list)
        self.z_list = []
        # Copy z-list with opposite sign for direct use in quspin
        for hz, site in z_list:
            self.z_list.append([-hz, site])
        self.basis = spin_basis_general(self.L)
        self.z_first = z_first

    @classmethod
    def from_coupling_list(
        cls, L: int, coupling_list: list[list], Jxx: float, Jyy: float, Jzz: float, hz: float, z_first: bool = False
    ):
        """Shortcut method for creating an instance from a coupling list.

        Parameters
        ----------
        L: int
            Number of spins.
        coupling_list: list
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jxx: float
            Coupling strength in X direction.
        Jyy: float
            Coupling strength in Y direction.
        Jzz: float
            Coupling strength in Z direction.
        hz : float
            Strength of magnatic field in Z direction.
        z_first : bool, optional
            Specifies whether the Z parts should be applied first in Trotterization.
            Default value is 'False'.

        Returns
        -------
        model: :class:`HeisenbergBonded`
            Heisneberg model with the specified bonds and parameters.
        """
        bonds_lists = []
        z_list = []
        parallel_lists = get_parallel_couplings(coupling_list)
        for coupling_list in parallel_lists:
            bond_list = []
            for i in range(len(coupling_list)):
                bond_list.append([Jxx, Jyy, Jzz, coupling_list[i][0], coupling_list[i][1]])
            bonds_lists.append(bond_list)
        for i in range(L):
            if hz != 0:
                z_list.append([hz, i])
        return cls(L, bonds_lists, z_list, z_first)

    @classmethod
    def from_lattice(
        cls, lattice: tenpy.models.lattice.Lattice, Jxx: float, Jyy: float, Jzz: float, hz: float, z_first: bool = False
    ):
        """Shortcut method for creating an instance from a lattice.

        Parameters
        ----------
        lattice: :class:`tenpy.models.lattice.Lattice`
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jxx: float
            Coupling strength in X direction.
        Jyy: float
            Coupling strength in Y direction.
        Jzz: float
            Coupling strength in Z direction.
        hz : float
            Strength of magnatic field in Z direction.
        z_first : bool, optional
            Specifies whether the Z parts should be applied first in Trotterization.
            Default value is 'False'.

        Returns
        -------
        model: :class:`HeisenbergBonded`
            Heisneberg model with the specified bonds and parameters.
        """
        coupling_list = get_coupling_list(lattice)
        L = len(lattice.order)
        return cls.from_coupling_list(L, coupling_list, Jxx, Jyy, Jzz, hz, z_first)

    @property
    def n_qbits(self) -> int:
        """Inherited property. see :py:attr:`Model.n_qbits`."""
        return self.L

    @property
    def n_terms(self) -> int:
        """Inherited property. see :py:attr:`Model.n_terms`."""
        if not self.z_list:
            return len(self.bonds_lists)
        else:
            return len(self.bonds_lists) + 1

    def _append_interaction_gates(self, qc, t, n_controls, bonds_list):
        if n_controls > 0:
            cnt_qbit = 0
            for _Jxx, _Jyy, _Jzz, qbit1, qbit2 in bonds_list:
                if _Jzz != 0:
                    append_CXXPhaseYYPhaseZZPhase(
                        qc,
                        2 / np.pi * t * _Jxx,
                        2 / np.pi * t * _Jyy,
                        2 / np.pi * t * _Jzz,
                        cnt_qbit,
                        qbit1 + n_controls,
                        qbit2 + n_controls,
                    )
                else:
                    append_CXXPhaseYYPhase(
                        qc, 2 / np.pi * t * _Jxx, 2 / np.pi * t * _Jyy, cnt_qbit, qbit1 + n_controls, qbit2 + n_controls
                    )

                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for _Jxx, _Jyy, _Jzz, qbit1, qbit2 in bonds_list:
                if _Jxx != 0:
                    qc.XXPhase(2 / np.pi * t * _Jxx, qbit1, qbit2)
                if _Jyy != 0:
                    qc.YYPhase(2 / np.pi * t * _Jyy, qbit1, qbit2)
                if _Jzz != 0:
                    qc.ZZPhase(2 / np.pi * t * _Jzz, qbit1, qbit2)

    def _append_Z_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for hz, qbit in self.z_list:
                if hz != 0:
                    append_CRz(qc, 2 / np.pi * t * hz, cnt_qbit, qbit + n_controls)
                    cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for hz, qbit in self.z_list:
                if hz != 0:
                    qc.Rz(2 / np.pi * t * hz, qbit)

    def append_gates(self, qc: pytket.circuit.Circuit, term: int, t: float, n_controls: int) -> None:
        """Inherited method. See :py:meth:`Model.append_gates`."""
        if term >= self.n_terms:
            raise ValueError("This model has only {} terms!".format(self.n_terms))

        if not self.z_list:
            self._append_interaction_gates(qc, t, n_controls, self.bonds_lists[term])
        else:
            if self.z_first:
                if term == 0:
                    self._append_Z_gates(qc, t, n_controls)
                else:
                    self._append_interaction_gates(qc, t, n_controls, self.bonds_lists[term - 1])
            else:
                if term == self.n_terms - 1:
                    self._append_Z_gates(qc, t, n_controls)
                else:
                    self._append_interaction_gates(qc, t, n_controls, self.bonds_lists[term])

    def _get_hamiltonian(self, ops, list):
        if list:
            return hamiltonian(
                [[ops, list]], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False
            )
        else:
            return hamiltonian([], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False)

    def _get_interaction_hamiltonian(self, bonds_list):
        xx_list = []
        yy_list = []
        zz_list = []
        for _Jxx, _Jyy, _Jzz, qbit1, qbit2 in bonds_list:
            if _Jxx != 0.0:
                xx_list.append([_Jxx, qbit1, qbit2])
            if _Jyy != 0.0:
                yy_list.append([_Jyy, qbit1, qbit2])
            if _Jzz != 0.0:
                zz_list.append([_Jzz, qbit1, qbit2])
        return (
            self._get_hamiltonian("xx", xx_list)
            + self._get_hamiltonian("yy", yy_list)
            + self._get_hamiltonian("zz", zz_list)
        )

    def _get_Z_hamiltonian(self):
        return self._get_hamiltonian("z", self.z_list)

    def get_hamiltonian(self, term: int) -> hamiltonian:
        """Inherited method. See :py:meth:`Model.get_hamiltonian`."""
        if term >= self.n_terms:
            raise ValueError("This model has only {} terms!".format(self.n_terms))

        if not self.z_list:
            return self._get_interaction_hamiltonian(self.bonds_lists[term])
        else:
            if self.z_first:
                if term == 0:
                    return self._get_Z_hamiltonian()
                else:
                    return self._get_interaction_hamiltonian(self.bonds_lists[term - 1])
            else:
                if term == self.n_terms - 1:
                    return self._get_Z_hamiltonian()
                else:
                    return self._get_interaction_hamiltonian(self.bonds_lists[term])


class HeisenbergXYZ(Model):
    """Class for handling Trotterized time evolution of a general Heisenberg model.
    Tortterization is done by seperating X, Y and Z terms giving in total 3 Trotter terms.
    For an alternative Trotterization, see :class:`HeisenbergBonded`.

    .. warning:: The Trotterization of this model does not conserve the total Sz symmetry even when Jx=Jy. 
                 In that case, use :class:`HeisenbergBonded` model instead.

    Parameters
    ----------
    L: int
        Number of spins
    xx_list: list
        List of lists specifying the XX interaction part of the hamiltonian.
        The first element of each list is a float representing the coupling strength.
        The other two elements are integers representing the interacting sites.
    yy_list: list
        List of lists specifying the YY interaction part of the hamiltonian.
        The first element of each list is a float representing the coupling strength.
        The other two elements are integers representing the interacting sites.
    zz_list: list
        List of lists specifying the ZZ interaction part of the hamiltonian.
        The first element of each list is a float representing the coupling strength.
        The other two elements are integers representing the interacting sites.
    z_list: list
        List of lists specifying the Z part of the hamiltonian.
        The first element of each list is a float representing the strength of magnatic field.
        The other element is an integer representing the site.
    order: str, optional
        Specifies whether the X or Y or Z parts should be applied first in Trotterization.
        Value can be any permutation of the string 'xyz'.
        Default value is 'xyz'.
    """

    def __init__(
        self,
        L: int,
        xx_list: list[list],
        yy_list: list[list],
        zz_list: list[list],
        z_list: list[list],
        order: str = "xyz",
    ):
        self.L = L
        # Copy lists with opposite sign for direct use in quspin
        self.xx_list = []
        for Jxx, site1, site2 in xx_list:
            self.xx_list.append([-Jxx, site1, site2])
        self.yy_list = []
        for Jyy, site1, site2 in yy_list:
            self.yy_list.append([-Jyy, site1, site2])
        self.zz_list = []
        for Jzz, site1, site2 in zz_list:
            self.zz_list.append([-Jzz, site1, site2])
        self.z_list = []
        for hz, site in z_list:
            self.z_list.append([-hz, site])
        self.basis = spin_basis_general(self.L)
        if sorted(order) != sorted("xyz"):
            raise ValueError("Unrecognized order!")
        else:
            self.order = order

    @classmethod
    def from_coupling_list(
        cls, L: int, coupling_list: list[list], Jxx: float, Jyy: float, Jzz: float, hz: float, order: str = "xyz"
    ):
        """Shortcut method for creating an instance from a coupling list.

        Parameters
        ----------
        L: int
            Number of spins.
        coupling_list: list
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jxx: float
            Coupling strength in X direction.
        Jyy: float
            Coupling strength in Y direction.
        Jzz: float
            Coupling strength in Z direction.
        hz : float
            Strength of magnatic field in Z direction.
        order: str, optional
            Specifies whether the X or Y or Z parts should be applied first in Trotterization.
            Value can be any permutation of the string 'xyz'.
            Default value is 'xyz'.

        Returns
        -------
        model: :class:`HeisenbergXYZ`
            Heisneberg model with the specified bonds and parameters.
        """
        xx_list = []
        yy_list = []
        zz_list = []
        z_list = []
        for i in range(len(coupling_list)):
            if Jxx != 0:
                xx_list.append([Jxx, coupling_list[i][0], coupling_list[i][1]])
            if Jyy != 0:
                yy_list.append([Jyy, coupling_list[i][0], coupling_list[i][1]])
            if Jzz != 0:
                zz_list.append([Jzz, coupling_list[i][0], coupling_list[i][1]])
        for i in range(L):
            if hz != 0:
                z_list.append([hz, i])
        return cls(L, xx_list, yy_list, zz_list, z_list, order)

    @classmethod
    def from_lattice(
        cls, lattice: tenpy.models.lattice.Lattice, Jxx: float, Jyy: float, Jzz: float, hz: float, order: str = "xyz"
    ):
        """Shortcut method for creating an instance from a lattice.

        Parameters
        ----------
        lattice: :class:`tenpy.models.lattice.Lattice`
            List of lists specifying the lattice bonds.
            Each list has two integers representing the sites of the bond.
        Jxx: float
            Coupling strength in X direction.
        Jyy: float
            Coupling strength in Y direction.
        Jzz: float
            Coupling strength in Z direction.
        hz : float
            Strength of magnatic field in Z direction.
        order: str, optional
            Specifies whether the X or Y or Z parts should be applied first in Trotterization.
            Value can be any permutation of the string 'xyz'.
            Default value is 'xyz'.

        Returns
        -------
        model: :class:`HeisenbergXYZ`
            Heisneberg model with the specified bonds and parameters.
        """
        coupling_list = get_coupling_list(lattice)
        L = len(lattice.order)
        return cls.from_coupling_list(L, coupling_list, Jxx, Jyy, Jzz, hz, order)

    @property
    def n_qbits(self) -> int:
        """Inherited property. see :py:attr:`Model.n_qbits`."""
        return self.L

    @property
    def n_terms(self) -> int:
        """Inherited property. see :py:attr:`Model.n_terms`."""
        return 3

    def _append_X_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for _Jxx, qbit1, qbit2 in self.xx_list:
                append_CXXPhase(qc, 2 / np.pi * t * _Jxx, cnt_qbit, qbit1 + n_controls, qbit2 + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for _Jxx, qbit1, qbit2 in self.xx_list:
                qc.XXPhase(2 / np.pi * t * _Jxx, qbit1, qbit2)

    def _append_Y_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for _Jyy, qbit1, qbit2 in self.yy_list:
                append_CYYPhase(qc, 2 / np.pi * t * _Jyy, cnt_qbit, qbit1 + n_controls, qbit2 + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for _Jyy, qbit1, qbit2 in self.yy_list:
                qc.YYPhase(2 / np.pi * t * _Jyy, qbit1, qbit2)

    def _append_Z_gates(self, qc, t, n_controls):
        if n_controls > 0:
            cnt_qbit = 0
            for _Jzz, qbit1, qbit2 in self.zz_list:
                append_CZZPhase(qc, 2 / np.pi * t * _Jzz, cnt_qbit, qbit1 + n_controls, qbit2 + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
            for hz, qbit in self.z_list:
                append_CRz(qc, 2 / np.pi * t * hz, cnt_qbit, qbit + n_controls)
                cnt_qbit = (cnt_qbit + 1) % n_controls
        else:
            for _Jzz, qbit1, qbit2 in self.zz_list:
                qc.ZZPhase(2 / np.pi * t * _Jzz, qbit1, qbit2)
            for hz, qbit in self.z_list:
                qc.Rz(2 / np.pi * t * hz, qbit)

    def append_gates(self, qc: pytket.circuit.Circuit, term: int, t: float, n_controls: int) -> None:
        """Inherited method. See :py:meth:`Model.append_gates`."""
        if term >= 3:
            raise ValueError("HeisenbergXYZ model has only 3 terms!")
        if self.order[term] == "x":
            self._append_X_gates(qc, t, n_controls)
        elif self.order[term] == "y":
            self._append_Y_gates(qc, t, n_controls)
        elif self.order[term] == "z":
            self._append_Z_gates(qc, t, n_controls)

    def _get_hamiltonian(self, ops, list):
        if list:
            return hamiltonian(
                [[ops, list]], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False
            )
        else:
            return hamiltonian([], [], basis=self.basis, dtype=np.float64, check_symm=False, check_herm=False)

    def _get_X_hamiltonian(self):
        return self._get_hamiltonian("xx", self.xx_list)

    def _get_Y_hamiltonian(self):
        return self._get_hamiltonian("yy", self.yy_list)

    def _get_Z_hamiltonian(self):
        return self._get_hamiltonian("zz", self.zz_list) + self._get_hamiltonian("z", self.z_list)

    def get_hamiltonian(self, term: int) -> hamiltonian:
        """Inherited method. See :py:meth:`Model.get_hamiltonian`."""
        if term >= 3:
            raise ValueError("HeisenbergXYZ model has only 3 terms!")
        if self.order[term] == "x":
            return self._get_X_hamiltonian()
        elif self.order[term] == "y":
            return self._get_Y_hamiltonian()
        elif self.order[term] == "z":
            return self._get_Z_hamiltonian()

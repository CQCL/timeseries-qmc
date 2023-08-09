from typing import Union
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import tenpy
from ..utils import get_coupling_list, get_kagome_triangles, get_adjacency_list
from ._sampler import Sampler


class ClusterUpdate(Sampler):
    """Sampler of spin product states using Wolff's cluster update for classical Ising models. For details of the
    algorithm, see *U. Wolff, Collective monte carlo updating for spin systems,* `Phys. Rev. Lett. 62, 361 (1989)
    <https://doi.org/10.1103/PhysRevLett.62.361>`_.

    Parameters
    ----------
    J: float
        Coupling strength.

    beta: float
        Inverse temperature.

    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
        Either this parameter or **lattice** needs to be specified.
        Both parameters cannot be specified simultaneously.

    lattice: :class:`tenpy.models.lattice.Lattice`
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
        Either this parameter or **coupling_list** needs to be specified.
        Both parameters cannot be specified simultaneously.
    """

    def __init__(
        self, J: float, beta: float, coupling_list: list[list] = None, lattice: tenpy.models.lattice.Lattice = None
    ):
        self.J = J
        self.T = 1.0 / beta
        if lattice and coupling_list:
            raise ValueError("'coupling_list' and 'lattice' parameters cannot be provided simultaneously!")
        if lattice:
            coupling_list = get_coupling_list(lattice)
        if coupling_list:
            self.adjacency_list = get_adjacency_list(coupling_list)
        else:
            raise ValueError("Either 'coupling_list' or 'lattice' parameters should be provided!")

    def propose_sample(self, old_sample: Union[list, np.ndarray], rng: np.random.RandomState):
        """Inherited method. See :py:meth:`Sampler.propose_sample`."""

        # start with random spin
        L = old_sample.size
        i = rng.randint(0, L)
        cluster = [i]
        # list of neighbours still to be traversed
        stack = [i]
        # while stack is not empty
        while stack:
            # take first spin in stack
            i = stack[0]
            for j in self.adjacency_list[i]:
                u = rng.rand()
                if i != j:
                    if (self.J > 0 and old_sample[i] == old_sample[j]) or (
                        self.J < 0 and old_sample[i] != old_sample[j]
                    ):
                        if j not in cluster:
                            # add spin to cluster and stack to be checked with prob. 1-exp(-2*beta*J)
                            if u < (1 - np.exp(-2 * np.abs(self.J) / self.T)):
                                stack.append(j)
                                cluster.append(j)
            # remove from stack
            stack.remove(i)
        # flip spins in cluster
        new_sample = np.copy(old_sample)
        for flip in cluster:
            new_sample[flip] += 1
            new_sample[flip] = new_sample[flip] % 2
        # calculate energy difference

        energydiff = 0
        mags_old = -(old_sample - 0.5) * 2
        mags_new = -(new_sample - 0.5) * 2
        for i in range(L):
            for j in self.adjacency_list[i]:
                energydiff += 0.5 * (mags_old[i] * mags_old[j] - mags_new[i] * mags_new[j])

        proposal_ratio = np.exp(self.J * energydiff / self.T)
        return new_sample, proposal_ratio


class KagomeClusterUpdate(Sampler):
    """Sampler of spin product states using cluster update for antiferromagnatic classical Ising model on
    Kagome lattices. For details of the algorithm, see *G. M. Zhang and C. Z. Yang, Cluster monte carlo dynamics for
    the antiferromagnetic ising model on a triangular lattice,* `Phys. Rev. B 50, 12546 (1994)
    <https://doi.org/10.1103/PhysRevB.50.12546>`_.

    .. note:: The lattice specified by **coupling_list** or **lattice** parameters need to be a kagome lattice.

    Parameters
    ----------
    J: float
        Coupling strength.

    beta: float
        Inverse temperature.

    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
        Either this parameter or **lattice** needs to be specified.
        Both parameters cannot be specified simultaneously.

    lattice: :class:`tenpy.models.lattice.Lattice`
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
        Either this parameter or **coupling_list** needs to be specified.
        Both parameters cannot be specified simultaneously.
    """

    def __init__(
        self, J: float, beta: float, coupling_list: list[list] = None, lattice: tenpy.models.lattice.Lattice = None
    ):
        if J > 0:
            raise ValueError(
                "'J' parameter should be negative. KagomeClusterUpdate works only for Anti-ferromagnatic models."
            )
        self.J = J
        self.T = 1.0 / beta
        if lattice and coupling_list:
            raise ValueError("'coupling_list' and 'lattice' parameters cannot be provided simultaneously!")
        if lattice:
            coupling_list = get_coupling_list(lattice)
        if coupling_list:
            self.bonds = list(coupling_list)
            self.bonds_in_triangle = get_kagome_triangles(coupling_list)
        else:
            raise ValueError("Either 'coupling_list' or 'lattice' parameters should be provided!")

    def _get_weights(self, state, rng):
        weights = np.zeros(len(self.bonds))
        spin = state * 2 - 1
        for t in range(len(self.bonds_in_triangle)):
            bonds_in_tr = np.zeros(3)
            for b in range(3):
                bonds_in_tr[b] = (
                    spin[self.bonds[self.bonds_in_triangle[t][b]][0]]
                    * spin[self.bonds[self.bonds_in_triangle[t][b]][1]]
                )
            energy = np.sum(bonds_in_tr)
            if energy == -1.0:
                u = rng.rand()
                if u < (1 - np.exp(-4 * np.abs(self.J) / self.T)):
                    # get satisfied bonds
                    args = np.argwhere(bonds_in_tr == -1.0)
                    v = rng.rand()
                    if v < 0.5:
                        weights[self.bonds_in_triangle[t][args[0][0]]] = 1.0
                    else:
                        weights[self.bonds_in_triangle[t][args[1][0]]] = 1.0
        return weights

    def _flip_spins(self, spins, N_components, labels, rng):
        spins = np.array(spins)
        flip_cluster = rng.random(N_components) < 0.5
        for n in range(spins.size):
            cluster = labels[n]
            if flip_cluster[cluster]:
                spins[n] = (spins[n] + 1) % 2
        return spins

    def propose_sample(self, old_sample: Union[list, np.ndarray], rng: np.random.RandomState):
        """Inherited method. See :py:meth:`Sampler.propose_sample`."""
        # start with random spin
        L = old_sample.size
        # Get which bonds are frozen
        weights = self._get_weights(old_sample, rng)
        # Find connected clusters
        graph = csr_matrix((weights, (np.array(self.bonds)[:, 0], np.array(self.bonds)[:, 1])), shape=(L, L))
        graph += csr_matrix((weights, (np.array(self.bonds)[:, 1], np.array(self.bonds)[:, 0])), shape=(L, L))
        N_components, labels = connected_components(graph, directed=False)
        # Flip clusters
        new_sample = self._flip_spins(old_sample, N_components, labels, rng)
        # calculate energy difference
        energydiff = 0
        mags_old = -(old_sample - 0.5) * 2
        mags_new = -(new_sample - 0.5) * 2
        energy_old = 0
        energy_new = 0
        for b in range(len(self.bonds)):
            energy_old += -mags_old[self.bonds[b][0]] * mags_old[self.bonds[b][1]]
            energy_new += -mags_new[self.bonds[b][0]] * mags_new[self.bonds[b][1]]
            energydiff += (
                mags_old[self.bonds[b][0]] * mags_old[self.bonds[b][1]]
                - mags_new[self.bonds[b][0]] * mags_new[self.bonds[b][1]]
            )
        proposal_ratio = np.exp(self.J * energydiff / self.T)
        return new_sample, proposal_ratio


import numpy as np
import tenpy

def get_coupling_list(lattice: tenpy.models.lattice.Lattice) -> list[list]:
    """Get the list of bonds between the different lattice sites.

    Parameters
    ----------
    lattice : :class:`tenpy.models.lattice.Lattice`

    Returns
    -------
    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
    """
    couplings = lattice.pairs['nearest_neighbors']
    qubit_couplings = []
    for u1, u2, dx in couplings:
        dx = np.r_[np.array(dx), u2 - u1]
        lat_idx_1 = lattice.order[lattice._mps_fix_u[u1], :]
        lat_idx_2 = lat_idx_1 + dx[np.newaxis, :]
        lat_idx_2_mod = np.column_stack((np.mod(lat_idx_2[:, :-1], lattice.Ls), np.full(lat_idx_2.shape[0], u2)))
        keep = lattice._keep_possible_couplings(lat_idx_2_mod[:, :-1], lat_idx_2[:, :-1], u2)

        sites1 = lat_idx_1[keep, :]
        sites2 = lat_idx_2_mod[keep, :]

        for s1, s2 in zip(sites1, sites2):
            qubit1 = np.where((lattice.order == s1).all(axis=1))[0][0]
            qubit2 = np.where((lattice.order == s2).all(axis=1))[0][0]
            qubit_couplings.append([qubit1, qubit2])
    return qubit_couplings

def _is_qubit_shared(coupling, coupling_list):
    for other in coupling_list:
        if(coupling[0] in other or coupling[1] in other):
            return True
    return False

def get_parallel_couplings(coupling_list: list[list]) -> list[list[list]]:
    """Seperate bonds into lists where no two bonds in the same list share a site.

    Parameters
    ----------
    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.
    """    
    to_process = list(coupling_list)
    parallel_lists = []
    while(to_process):
        new_list = []
        for coupling in to_process:
            if(not _is_qubit_shared(coupling, new_list)):
                new_list.append(coupling)
        to_process = [x for x in to_process if x not in new_list]
        parallel_lists.append(new_list)
    return parallel_lists

def get_adjacency_list(coupling_list: list[list]) -> list[list]:
    """Return a list of nearest neighbors for each site of the lattice.

    Parameters
    ----------
    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.

    Returns
    -------
    neighbors_list: list
        List of lists specifying the neighbors.
        The i-th list contains all neighbors of site i.
    """

    #Find largest index
    L = max([i for sublist in coupling_list for i in sublist])
    # Find neighbours of j
    adj_list = []
    for j in range(L+1):
        neighbours_of_j = []
        for interaction in coupling_list:
            if j in interaction:
                neighbours_of_j.extend([x for x in interaction if x != j])

        adj_list.append(neighbours_of_j)
    return adj_list

def get_kagome_triangles(coupling_list: list[list]) -> list[list]:
    """Return a list of all triangles in a kagome lattice.

    Parameters
    ----------
    coupling_list: list
        List of lists specifying the lattice bonds.
        Each list has two integers representing the sites of the bond.

    Returns
    -------
    triangles_list: list
        List of lists specifying the triangles.
        Each list has three intergers represnting the vertices of the triangle.
    """        
    coupling_list2 = list(coupling_list)
    triangles = []
    while(coupling_list2):
        first_bond = coupling_list2.pop(0)
        second_bond = None
        third_bond = None
        for i, bond in enumerate(coupling_list2):
            if(bond[0] in first_bond or bond[1] in first_bond):
                second_bond = bond
                del coupling_list2[i]
                break
        if(not second_bond):
            raise ValueError("Something went wrong. Check that coupling_list represents a kagome lattice!")
        third_bond_vertices = (set(first_bond).union(set(second_bond)))-set(first_bond).intersection(set(second_bond))
        for i, bond in enumerate(coupling_list2):
            if(bond[0] in third_bond_vertices and bond[1] in third_bond_vertices):
                third_bond = bond
                del coupling_list2[i]
                break
        if(not third_bond):
            raise ValueError("Something went wrong. Check that coupling_list represents a kagome lattice!")
        triangle = [coupling_list.index(first_bond), coupling_list.index(second_bond), coupling_list.index(third_bond)]
        triangles.append(triangle)
    return triangles
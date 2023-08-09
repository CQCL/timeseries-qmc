from typing import Union
import numpy as np
from pytket.circuit import Circuit

def int_to_bitstring(value: int, n_bits: int) -> np.ndarray:
    """ Converts an integer value to an array of bits (ones and zeros).

    Parameters
    ----------
    val: int
        Integer value.

    n_bits: int
        Minimum number of bits that is returned.
        If actual number of necessary bits is less than this number, the list is padded with zero bits.
    """
    bits = [int(digit) for digit in bin(value)[2:]]
    while(len(bits)<n_bits):
        bits.insert(0,0)
    return np.array(bits)

def bitstring_to_int(bitstring: Union[list, np.ndarray]) -> int:
    """ Returns the integer represented by a bitstring.

    Parameters
    ----------
    psi_bitstring: list or :py:class:`numpy.ndarray`
        List/array of zeros and ones.    
    """
    return int(np.array2string(np.array(bitstring), separator='')[1:-1],2)

def bitstring_to_state(bitstring: Union[list, np.ndarray]) -> np.ndarray:
    """Return an array that contains all zeros except a single one located at the index 
    specified by a bitstring. For example, for input ``[0,0]`` the output is ``np.array([1,0,0,0])``,
    and for input ``[0,1]`` the output is ``np.array([0,0,1,0])``.

    Parameters
    ----------
    psi_bitstring: list or :py:class:`numpy.ndarray`
        List/array of zeros and ones.
    """
    L = len(bitstring)
    index_that_is_1 = bitstring_to_int(bitstring)
    psi = np.zeros(2**L)
    psi[index_that_is_1] = 1
    return psi

def bitstring_to_circuit(bitstring: Union[list, np.ndarray]) -> Circuit:
    r"""Return a :py:class:`pytket.Circuit <pytket._tket.circuit.Circuit>` that transforms the :math:`|0\dots0\rangle` 
    state into the product state specified by a bitstring.

    Parameters
    ----------
    psi_bitstring: list or :py:class:`numpy.ndarray`
        List/array of zeros and ones.
    """
    L = len(bitstring)
    circuit = Circuit(L)
    for i in range(L):
        if(bitstring[i]):
            circuit.X(i)
    return circuit

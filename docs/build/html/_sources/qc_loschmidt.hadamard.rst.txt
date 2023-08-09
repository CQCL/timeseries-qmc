QC Loschmidt Echos - Hadamard Test
****************************************

This example shows how to estimate the energy of Heisenberg model on 12-sites kagome lattice with periodic boundaries 
using cluster update with Boltzmann weights estimated from noisy Loschmidt echos with non-negative least squares method. 
The Loschmidt echos are estimated via Hadarmard test quantum circuits.

Script
======

:download:`download script <../../examples/qc_loschmidt_hadamard.py>`

.. literalinclude:: ../../examples/qc_loschmidt_hadamard.py
   :language: python

Output
======
**Note:** Your output might be different due to the stastical nature of the algorithm.

.. image:: ../../examples/data/qc_loschmidt_hadamard/output.png
  :alt: Output image
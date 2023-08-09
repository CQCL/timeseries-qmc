Exact Loschmidt Echos - Fourier Transform
******************************************

This example shows how to run the time-seriers QMC algorithm for  transverse filed Ising model on 10-sites
honeycomb lattice with open boundaries using cluster update with Boltzmann weights estimated from exact Loschmidt
echos with fourier trasform + Gaussian filer.

.. note::
   The number of time points and maximum time were intentionally set to small values to illustrate the difference with
   :ref:`the non-negative least squares method <nnls>`.
   Better results (i.e. smaller bias) can be obtained by increasing the number of time points and maximum time. 

Script
======

:download:`download script <../../examples/exact_loschmidt_fourier.py>`

.. literalinclude:: ../../examples/exact_loschmidt_fourier.py
   :language: python

Output
======
**Note:** Your output might be different due to the stastical nature of the algorithm.

.. image:: ../../examples/data/exact_loschmidt_fourier/output.png
  :alt: Output image
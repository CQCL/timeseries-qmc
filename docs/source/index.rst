.. TimeseriesQMC documentation master file, created by
   sphinx-quickstart on Fri Jul 14 14:14:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TimeseriesQMC's documentation!
*****************************************
This is a python library for running and testing the time-seriers quantum Monte Carlo algorithm of model Hamiltonians.
Time-series quantum Monte Carlo is a hybrid classical-quantum algorithm for evaluting finite-temperature obervables 
by sampling easily preparable states and estimating their thermal weights from real-time dynamics simulated on a quantum
computer. :footcite:`Lu2021, Schuckert2022, Ghanem2023`

The source code is avaiable on `Github <https://github.com/CQCL/timeseries-qmc>`_.

How to Cite
===========
If you use this library for a work published in an academic journal, we apperciate citing the following paper to 
acknowledge the work put into the development:

..  code-block:: bibtex

  @misc{Ghanem2023,
   title={Robust Extraction of Thermal Observables from State Sampling and Real-Time Dynamics on Quantum Computers}, 
   author={Khaldoon Ghanem and Alexander Schuckert and Henrik Dreyer},
   year={2023},
   eprint={2305.19322},
   archivePrefix={arXiv},
   primaryClass={quant-ph}
  }

Development Team
================
This library was developed by the following people at the condesend matter group of `Quantinuum 
<https://www.quantinuum.com/>`_:
 
* Khaldoon Ghanem (khaldoon.ghanem@quantinuum.com)
* Alexander Schucker
* Kevin Hemery
* Ramil Nigmatullin
* Mohsin Iqbal
* Ella Crane
* Henrik Dreyer (henrik.dreyer@quantinuum.com)


Acknowledgment
==============
This work was supported by the German Federal Ministry of Education and Research (BMBF) through the project `EQUAHUMO 
<https://www.quantentechnologien.de/forschung/foerderung/anwendungsnetzwerk-fuer-das-quantencomputing/equahumo.html>`_
(grant number 13N16069) within the funding program quantum technologies - from basic research to market.


Licence
=======
The code is licensed under `Apache License Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0.txt>`_.

References
==========
.. footbibliography::

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   Home <self> 
   install
   tutorial
   examples
   api
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

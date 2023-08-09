Installation
************


The timeseries-qmc package depends on the following packages:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `pytket <https://cqcl.github.io/tket/pytket/api/>`_
* `pytket-qiskit <https://cqcl.github.io/pytket-qiskit/api/index.html>`_
* `physics-tenpy <https://tenpy.readthedocs.io/en/latest/>`_
* `quspin <https://quspin.github.io/QuSpin/index.html>`_

The timeseries-qmc package and all its dependencies, except **quspin**, are avaiable via pip and will be 
installed automatically by executing the following command:

.. code-block:: shell

    pip install timeseries-qmc

quspin needs to be installed seperately, either manually or via Anaconda. 
For details on how to intall it, please refere to its `installation guide <https://quspin.github.io/QuSpin/Installation.html>`_
after reading the following section.

Installing QuSpin via Anoaconda
================================
It is known that pip and conda do not play well together (see
this `page <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_ on the topic of using pip in conda 
enviroments).
Therefore, it is recommonded to install quspin  manually.
However, installing a package manually can be cumbersome and you may still opt for installing quspin via Anaconda.
In that case, it is recommonded to first setup a new conda environment,
install all conda packages (including quspin), and then install all pip packages (including timeseries-qmc).
The following is a minimal example:

.. code-block:: shell

    conda create --name myenv python=3.9
    conda activate myenv 
    conda install -c weinbe58 quspin 
    pip install timeseries-qmc

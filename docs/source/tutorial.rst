Tutorial
########

Models
======
The starting point for any calculations is specifying the model Hamiltonian via one of the 
:ref:`Model classes <models>`.
For exampling, one can create a Transverse-field Ising model on a 4-sites ring as following:

.. jupyter-execute::

    import timeseries_qmc as ts

    L = 4 #Number of sites
    coupuling_list = [[0, 1], [1, 2], [2, 3], [3, 0]] # Connectivity of sites
    Jzz = 1.0 # Strength of Coupling
    hx = 1.0 # Strength of field
    model = ts.models.Ising.from_coupling_list(L, coupuling_list, Jzz,  hx)

The parameter ``coupuling_list`` is a list of lists specifying how the lattice sites are connected.
Alternatively, one can use :class:`tenpy.models.lattice.Lattice` as a convenient shortcut. 
For example, the following code creates a 3x3 square lattice:

.. jupyter-execute::

    import timeseries_qmc as ts
    from tenpy.models import lattice

    lattice = lattice.Square(3, 3, None, bc='periodic')
    Jzz = 1.0 # Strength of Coupling
    hx = 1.0 # Strength of field
    model = ts.models.Ising.from_lattice(lattice, Jzz, hx)

The :class:`Model <timeseries_qmc.models.Model>` class is the basic building block and contains several methods that are 
employed by other classes in the package.
For the sake of benchmarking later, you can directly calculate some thermal observables via exact diagonalization when 
the system size is small

.. jupyter-execute::

    beta = 0.3 # Inverse temperature
    M2, E, E2 = model.calc_thermal_observables(beta) # Calculates: Manetization^2, Energy, Energy^2

    Cv=(E2-E**2)*beta**2 # Specific heat
    print("Magnetizaiton: {:.3f}".format(M2))
    print("Specific Heat: {:.3f}".format(Cv))

Loschmidt Echos
===============
Given a model, we next would like to evalute the Loschmidt echos of its product states.
This is achieved via one of the :ref:`Loschmidt echo evaluators <loschmidt>`.
The most accurate would be to use :class:`.ExactLoschmidtEcho`.
This performs the exact time evoluation using `QuSpin package <https://quspin.github.io/QuSpin/>`_ under the hood.
One can also check the effect of Trotterization by using :class:`.TrotterizedLoschmidtEcho`.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pylab as plt

    n_t = 20 # Number of time points
    t_max = 1.0 # Maximum time
    dt = t_max/n_t
    t = np.linspace(dt, t_max, n_t, endpoint=True) # Points at which to evaluate Loschmidt echos
    psi =  model.n_qbits*[0] # Product state for which to evaluate Loschmidt echos

    exact_loschmidt = ts.loschmidt.ExactLoschmidtEcho(model)
    trotterized_loschmidt = ts.loschmidt.TrotterizedLoschmidtEcho(model, trotter_order=1, dt_trotter=0.25)

    G_exact,_  = exact_loschmidt.evaluate(psi, t)
    G_trotter,_  = trotterized_loschmidt.evaluate(psi, t)

    plt.plot(t, G_exact.real, label="Exact")
    plt.plot(t, G_trotter.real, label="Trotterized")
    plt.xlabel("t")
    plt.ylabel("Re[G(t)]")
    plt.legend()
    plt.show()

    plt.plot(t, G_exact.imag, label="Exact")
    plt.plot(t, G_trotter.imag, label="Trotterized")
    plt.xlabel("t")
    plt.ylabel("Im[G(t)]")
    plt.legend()
    plt.show()

More interestingly and more relevant would be to evaluate Loschmidt echo using quantum circuits.
There are currently two evaluators: :class:`.Hadamard` and 
:class:`.Catstate`. 
The first uses the `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`_ and is 
applicable to any model, while the second uses the Catstate trick :footcite:`Lu2021` and is applicable to models
where the product state :math:`|0\dots0\rangle` is an eigenstate of the Trotterized Hamiltonian e.g.
:class:`.HeisenbergBonded`. Both these evaluators need a :class:`pytket.backends.backend.Backend` for running
the quantum circuits. Optionally, you can aslo provide via the parameter ``cache_directroy`` a path for a folder where 
compiled circuits, job handles and shots are stored. This is useful for saving intermidiate results and avoid rerunning
costly circuits.

.. jupyter-execute::

    from pytket.extensions.qiskit import AerBackend

    backend = AerBackend()
    hadmard_loschmidt = ts.loschmidt.Hadamard(model, backend, n_shots = 256, cache_directory='tutorial_data', dt_trotter=0.25) 
    G_aer, G_aer_err  = hadmard_loschmidt.evaluate(psi, t)

    plt.plot(t, G_exact.real, label="Exact")
    plt.plot(t, G_trotter.real, label="Trotterized")
    plt.errorbar(t, G_aer.real, yerr=G_aer_err.real, label="Aer")
    plt.xlabel("t")
    plt.ylabel("Re[G(t)]")
    plt.legend()
    plt.show()


Boltzmann Weights
=================
Calculating Boltzmann weights (and the local energy moments) is done using 
:ref:`Boltzmann weight calculators <boltzmann>`.
The two calculators :class:`.FourierTransform` and :class:`.NNLS` allows extracting these quanties from a time series of 
Loschmidt echos via Wick's rotation. :footcite:`Ghanem2023`
Additionally, for benchmarking purposes, :class:`ExactBoltzmannWeight` allows calculating these quantities exactly 
via QuSpin.
The use of :class:`.NNLS` is recommended.

.. jupyter-execute::

    exact_boltzmann = ts.boltzmann.ExactBoltzmannWeight(model, beta)
    fourier_boltzmann = ts.boltzmann.FourierTransform(beta, exact_loschmidt, t_max=1.0, n_t = 20)
    nnls_boltzmann = ts.boltzmann.NNLS(model, beta, exact_loschmidt, t_max=1.0, n_t = 20)

    exact_weight, exact_e, exact_e2 = exact_boltzmann.calculate(psi)
    fourier_weight, fourier_e, fourier_e2 = fourier_boltzmann.calculate(psi)
    nnls_weight, nnls_e, nnls_e2 = nnls_boltzmann.calculate(psi)

    print("Boltzmann Weight - Exact   : {:.3f}".format(exact_weight))
    print("Boltzmann Weight - Fourier : {:.3f}".format(fourier_weight))
    print("Boltzmann Weight - NNLS    : {:.3f}".format(nnls_weight))


Monte Carlo Sampling
====================
In order to generate samples via time-series QMC, we need a :class:`.MarkovChain` object.
This requries a :class:`.BoltzmannWeightCalculator`, which was disucssed earlier, and one of the
:ref:`samplers <sampling>`, which is used to propose samples.
The efficeincy/correlation of the Markov chain will depend on the choosen sampler.
The use of :class:`.ClusterUpdate` is recommended over :class:`.SingleFlip`.
For anti-ferromagnetic models on Kagome lattices, :class:`.kagomeClusterUpdate` is recommended.

.. jupyter-execute::

    sampler = ts.sampling.ClusterUpdate(Jzz, beta, lattice=lattice)
    rng_seed = 32 # Seed for random number generation
    initial_sample = psi # Starting state 
    chain = ts.sampling.MarkovChain(exact_boltzmann, sampler, initial_sample, rng_seed)
    n_samples = 100

    for i in range(n_samples):
        chain.generate_next_sample()
        if((i+1)%10==0):
            print("Sample #{}: {}".format(i+1, chain.current_sample))

    mag_vals = chain.get_magnetziations_squared()
    print("")
    print("Average Magnetizaiton: {:.3f}".format(np.mean(mag_vals)))

:class:`.MarkovChain` class allows saving itself to a file. This can be done at the end of sampling or periodically in 
order to be able to retreive partial results when something goes wrong.
A simpler approach would be to use the function :func:`generate_chain`.
The functions takes as input the desrired number of samples and the path where the chain is stored.
The chain is attempted to be loaded from the file and the sampling continues till the specified number of samples is 
reached. Also the chain is automatically saved after each sample is generated.

.. jupyter-execute::

    import math

    chains = []
    samples_num_per_chain = 1000
    chains_num = 4
    for rng_seed in range(chains_num):
        chain_filename = "./tutorial_data/chain_"+str(rng_seed)+".pickle"
        chain = ts.sampling.generate_chain(exact_boltzmann, sampler, initial_sample, 
                                rng_seed, samples_num_per_chain, chain_filename)
        chains.append(chain)

    num_burned_samples = 50
    used_samples = list(range(num_burned_samples, samples_num_per_chain))
    chains_estimates = []
    for chain in chains:
        values = chain.get_magnetziations_squared()[used_samples]
        cum_means = np.cumsum(values)/np.arange(1, 1+len(values))
        plt.plot(used_samples, cum_means, 'o', ms=2)
        chain_estimate = np.mean(values)
        chains_estimates.append(chain_estimate)

    chains_estimates = np.array(chains_estimates)
    estimate = np.mean(chains_estimates)
    error = 2*np.std(chains_estimates)/math.sqrt(chains_num) #Two standard deviations - 95% condfidence

    plt.hlines(xmin=used_samples[0], xmax=used_samples[-1], y=M2, color="k", ls="--", label="Exact")
    plt.hlines(xmin=used_samples[0], xmax=used_samples[-1], y=estimate,color="k", label="Estimate")
    plt.fill_between(used_samples, estimate-error, estimate+error, color='k', zorder=2, alpha=0.5, label="95% Confidence")
    plt.ylabel("$M^2$")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()


Error Mitigation
================
When obtaining Loschmidt echos by running quantum circuits on real quantum computers, hardware noise is inevitable.
We provide two simple error mitgation stratigies, depending on the way Loschmidt echos are calculated.
For the Hadamard test, we mitigate the effect of depolarizing noise by rescaling the measured Loschmidt echos by
a factor related to the fidelity of the quantum gates and the number of those gates in the circuit.
This will be done automatically by passing :class:`.FidelityEstimator` as a parameter to  :class:`.Hadamard`.
For the catestate trick, we mitigate the effect of noise by filter the shots which violate the symmetry of the 
Hamiltonian. Filtering these shots is achieved by passing :class:`.SymmetryFilter` as a parameter to :class:`.Catstate`.


Log Messages
============
TimeseriesQMC package print some log messages that can be helpful to keep track of the current status of sampling and
the compiliation and running of quantum circuits. To activate logging simply call the function 
:func:`.configure_logging` which accepts as parameters the path to the log file and whetehr log messages should also be
printted to the standard output.

References
==========
.. footbibliography::
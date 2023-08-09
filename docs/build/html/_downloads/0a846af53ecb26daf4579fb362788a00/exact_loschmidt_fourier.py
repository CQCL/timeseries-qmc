import math
import os
import timeseries_qmc as ts
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

experiment_dir = "./data/exact_loschmidt_fourier"
log_filename = os.path.join(experiment_dir, "log")
ts.utils.configure_logging(log_filename, True)

#############
# Setup Model
#############
# 10-sites honeycomb lattice
L = 10
coupling_list = [[0, 1], [2, 3], [5, 6], [7, 8], [1, 2], [3, 4], [6, 7], [8, 9], [0, 9], [2, 7], [4, 5]]
Jzz = 1
hx = 1.0
model = ts.models.Ising.from_coupling_list(L, coupling_list, Jzz, hx)

#############################
# Perform Monte Carlo Samping
#############################
beta = math.log(2 + math.sqrt(3)) / 2  # Critical temperature of classical model
loschmidt_evaluator = ts.loschmidt.ExactLoschmidtEcho(model)
boltzmann_calculator = ts.boltzmann.FourierTransform(beta, loschmidt_evaluator, t_max=0.5, n_t=5)
sampler = ts.sampling.ClusterUpdate(Jzz, beta, coupling_list=coupling_list)
samples_num_per_chain = 128
chains_num = 4

initial_state = model.n_qbits * [0]
chains = []
for rng_seed in range(chains_num):
    chain_filename = os.path.join(experiment_dir, "chain_" + str(rng_seed) + ".pickle")
    chain = ts.sampling.generate_chain(
        boltzmann_calculator, sampler, initial_state, rng_seed, samples_num_per_chain, chain_filename
    )
    chains.append(chain)

#############################
# Estimation & Visualization
#############################
exact, _, _ = model.calc_thermal_observables(beta)

num_burned_samples = 16
used_samples = list(range(num_burned_samples, samples_num_per_chain))
chains_estimates = []

fig = plt.figure()
for chain in chains:
    values = chain.get_magnetziations_squared()[used_samples]
    cum_means = np.cumsum(values) / np.arange(1, 1 + len(values))
    plt.plot(used_samples, cum_means, "o", ms=2)
    chain_estimate = np.mean(values)
    chains_estimates.append(chain_estimate)

chains_estimates = np.array(chains_estimates)
estimate = np.mean(chains_estimates)
error = 2 * np.std(chains_estimates) / math.sqrt(chains_num)  # Two standard deviations - 95% condfidence

plt.hlines(xmin=used_samples[0], xmax=used_samples[-1], y=exact, color="k", ls="--", label="Exact")
plt.hlines(xmin=used_samples[0], xmax=used_samples[-1], y=estimate, color="k", label="Estimate")
plt.fill_between(
    used_samples, estimate - error, estimate + error, color="k", zorder=2, alpha=0.5, label="95% Confidence"
)
plt.ylabel("$M^2$")
plt.xlabel("Iteration")
plt.legend()
plt.savefig(os.path.join(experiment_dir, "output.png"), bbox_inches="tight")
plt.show()

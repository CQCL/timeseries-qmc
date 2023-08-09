from typing import Union
import pickle
import shutil
import os
import numpy as np
import logging
from ..boltzmann import BoltzmannWeightCalculator
from ..utils import bitstring_to_int
from ._sampler import Sampler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MarkovChain:
    """Class representing a Markov chain of time-series quantum Monte Carlo samples.

    Parameters
    ----------
    boltzmann_weight_calculator: :class:`.boltzmann.BoltzmannWeightCalculator`
        Calculator of Boltzmann weights.

    sampler: :class:`Sampler`
        Sampler of quantum states.

    initia_sample_bitstring: list or :py:class:`numpy.ndarray`
        List/array of zeros and ones representing initial product state.

    seed: int
        Seed for the random number generator.
    """

    def __init__(
        self,
        boltzmann_weight_calculator: BoltzmannWeightCalculator,
        sampler: Sampler,
        inital_sample_bitstring: Union[list, np.ndarray],
        seed: int,
    ):
        self.boltzmann_weight_calculator = boltzmann_weight_calculator
        self.sampler = sampler
        self._samples = []
        self._w = {}  # Boltzmann weight <\psi|e^{-\beta H}|\psi>

        self._e = {}  # Local energy <\psi|e^{-\beta H} H|\psi>/<\psi|e^{-\beta H}|\psi>
        self._e_sq = {}  # Local squared energy <\psi|e^{-\beta H} H^2|\psi>/<\psi|e^{-\beta H}|\psi>

        self._accepted_samples_count = 0
        self._current_sample = np.array(inital_sample_bitstring)
        self._random = np.random.RandomState(seed)

    @property
    def current_sample(self) -> np.ndarray:
        """Bitstring of zeros and ones representing last generated product state."""
        view = self._current_sample.view()
        view.flags.writeable = False
        return view

    def get_samples(self) -> list:
        """Gets a copy of all samples of the chain.

        Returns
        -------
        samples: list
            List of bitstrings of zeros and ones representing the generated product states.
            The elements are ordered by their generation time.
        """
        vals = []
        for sample in self._samples:
            val = np.array(sample)
            vals.append(val)
        return np.array(vals)

    def get_magnetziations_squared(self) -> list:
        """Calculates the squared magnetization for all samples of the chain.

        Returns
        -------
        values: list
            List of numbers represnting the squared magnetizations.
            The elements are ordered by their generation time.
        """
        vals = []
        for sample in self._samples:
            val = np.mean(1 - 2 * sample) ** 2
            vals.append(val)
        return np.array(vals)

    def get_boltzmann_weights(self):
        """Gets the Boltzmann weights for all samples of the chain.

        Returns
        -------
        values: list
            List of numbers represnting the Boltzmann weights.
            The elements are ordered by their generation time.
        """
        vals = []
        for sample in self._samples:
            sample_int = bitstring_to_int(sample)
            val = self._w[sample_int]
            vals.append(val)
        return np.array(vals)

    def get_local_energies(self):
        """Gets the local energies for all samples of the chain.

        Returns
        -------
        values: list
            List of numbers represnting the local energies.
            The elements are ordered by their generation time.
        """
        vals = []
        for sample in self._samples:
            sample_int = bitstring_to_int(sample)
            val = self._e[sample_int]
            vals.append(val)
        return np.array(vals)

    def get_local_energies_squared(self):
        """Gets the second moment of local energies for all samples of the chain.

        Returns
        -------
        values: list
            List of numbers represnting the second moment of local energies.
            The elements are ordered by their generation time.
        """
        vals = []
        for sample in self._samples:
            sample_int = bitstring_to_int(sample)
            val = self._e_sq[sample_int]
            vals.append(val)
        return np.array(vals)

    def generate_next_sample(self):
        """Adds a new sample to the chain."""
        new_sample, proposal_ratio = self.sampler.propose_sample(self._current_sample, self._random)
        u = self._random.rand()
        alpha = (self._get_probability(new_sample) / self._get_probability(self._current_sample)) * proposal_ratio
        if u <= alpha:
            logger.info(
                "Transition{}: {} -> {}, alpha = {:.3f}, ACCEPTED".format(
                    self.generated_samples_count,
                    bitstring_to_int(self._current_sample),
                    bitstring_to_int(new_sample),
                    alpha,
                )
            )
            self._current_sample = new_sample
            self._accepted_samples_count += 1
        else:
            logger.info(
                "Transition{}: {} -> {}, alpha = {:.3f}, REJECTED".format(
                    self.generated_samples_count,
                    bitstring_to_int(self._current_sample),
                    bitstring_to_int(new_sample),
                    alpha,
                )
            )

        self._samples.append(self._current_sample)

    def _get_probability(self, sample):
        sample_int = bitstring_to_int(sample)
        w_sample = self._w.get(sample_int)
        if w_sample is None:
            w_sample, e_sample, e_sq_sample = self.boltzmann_weight_calculator.calculate(sample)
            self._e[sample_int] = e_sample
            self._e_sq[sample_int] = e_sq_sample

            self._w[sample_int] = w_sample

        return w_sample

    @property
    def generated_samples_count(self) -> int:
        """Total number of samples generated sofar."""
        return len(self._samples)

    @property
    def distinct_samples_count(self) -> int:
        """Total number of *unique* samples generated sofar."""
        return len(self._w)

    @property
    def acceptance_rate(self) -> int:
        """Acceptance ratio of proposed samples.

        .. note:: This value is ``None`` when no sample has been generated so far.
        """
        if self.generated_samples_count > 0:
            return self._accepted_samples_count / len(self._samples)
        else:
            return None

    @property
    def distinct_samples_ratio(self) -> int:
        """Ratio of distinct samples to the total number of generated samples."""
        return self.distinct_samples_count / (self.generated_samples_count + 1)  # +1 is for initial sample

    def load(self, file_path: str):
        """Load a chain of samples from file. It overwrites the content of the chain with the loaded one."""
        with open(file_path, "rb") as f:
            (
                self._samples,
                self._w,
                self._accepted_samples_count,
                self._current_sample,
                self._random,
                self._e,
                self._e_sq,
            ) = pickle.load(f)

    def save(self, file_path: str):
        """Save a chain of samples to a file. It creates parent directories if necessary."""
        backup_made = False
        if os.path.isfile(file_path):
            backup_filename = file_path + ".backup"
            shutil.copyfile(file_path, backup_filename)
            backup_made = True
        else:
            dir = os.path.dirname(file_path) # Create parent directories if necessary.
            if(dir):
                os.makedirs(dir, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(
                (
                    self._samples,
                    self._w,
                    self._accepted_samples_count,
                    self._current_sample,
                    self._random,
                    self._e,
                    self._e_sq,
                ),
                f,
            )
        if backup_made:
            os.remove(backup_filename)


def generate_chain(
    boltzmann_weight_calculator: BoltzmannWeightCalculator,
    sampler: Sampler,
    inital_sample_bitstring: Union[list, np.ndarray],
    seed: int,
    num_samples: int,
    chain_file_path=None,
):
    """Generate a Markov chain with a specified number of samples.

    If parameter **chain_file_path** is supplied, the chain is attempted to be loaded from the file and the sampling 
    continues till the specified number of samples is reached. Also the chain is saved after each sample is generated.

    Parameters
    ----------
    boltzmann_weight_calculator: :class:`.boltzmann.BoltzmannWeightCalculator`
        Calculator of Boltzmann weights.

    sampler: :class:`Sampler`
        Sampler of quantum states.

    initia_sample_bitstring: list or :py:class:`numpy.ndarray`
        List/array of zeros and ones representing initial product state.

    seed: int
        Seed for the random number generator.

    num_samples: int
        Number of samples in the chain.

    chain_file_path: str
        Path of the file used to store/retrieve the chain data.
        Value ``None`` can be used to avoid storing/retreiving data.

    Returns
    -------
    chain: MarkovChain
        Markov chain with the specified number of samples.
    """
    chain = MarkovChain(boltzmann_weight_calculator, sampler, inital_sample_bitstring, seed)
    if chain_file_path:
        try:
            chain.load(chain_file_path)
            logger.info("Chain_{} with {} samples has been found.".format(seed, chain.generated_samples_count))
            logger.info("Resuming sampling...")
        except IOError:
            logger.info("Starting sampling Chain_{}...".format(seed))
            chain.save(chain_file_path)
    else:
        logger.info("Starting sampling Chain_{}...".format(seed))

    while chain.generated_samples_count < num_samples:
        chain.generate_next_sample()
        if chain_file_path:
            chain.save(chain_file_path)

    logger.info("Sampling is Done!")
    return chain

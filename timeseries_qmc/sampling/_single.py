from typing import Union
import numpy as np
from ._sampler import Sampler


class SingleFlip(Sampler):
    """Sampler of spin product states using single spin flips."""

    def propose_sample(self, old_sample: Union[list, np.ndarray], rng: np.random.RandomState):
        """Inherited method. See :py:meth:`Sampler.propose_sample`."""
        L = old_sample.size
        new_sample = np.array(old_sample)
        i = rng.randint(L)
        new_sample[i] += 1
        new_sample[i] = new_sample[i] % 2
        return new_sample, 1.0

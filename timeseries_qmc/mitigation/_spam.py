import numpy as np


def correct_measurment_error(p0: float, m0: float = 0.999, m1: float = 0.995) -> float:
    """Corrects the measurement error of a single qubit.

    Parameters
    ----------
    p0: float
        The measured probability of 0.

    m0: float
        The probability of measuring 0 when the actual value is 0.

    m1: float
        The probability of measuring 1 when the actual value is 1.

    Returns
    -------
    p0_corrected: float
        The corrected probability of measuring 0.
    """
    M = np.array([[m0, 1 - m1], [1 - m0, m1]])
    M_inv = np.linalg.inv(M)
    p0_corrected, _ = M_inv @ np.array([p0, 1 - p0])
    return p0_corrected

"""
This module defines general-purpose objects, functions and classes.

"""

import numpy as np


def get_tensor_eigenvalues(T, sort=False):
    if sort:
        return (sorted(np.linalg.eigvals(T), reverse=True))
    else:
        return (np.linalg.eigvals(T))

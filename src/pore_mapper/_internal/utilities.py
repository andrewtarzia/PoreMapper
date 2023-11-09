import numpy as np


def get_tensor_eigenvalues(
    T: np.ndarray, sort: bool = False
) -> list | np.ndarray:
    if sort:
        return sorted(np.linalg.eigvals(T), reverse=True)
    else:
        return np.linalg.eigvals(T)

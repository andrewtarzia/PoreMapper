import pytest
from pore_mapper import get_tensor_eigenvalues
import numpy as np


@pytest.fixture
def tensor():
    return np.array([
        [-0.09852765,  0.633203,   -0.05770473],
        [ 0.76346367, -0.09852765,  0.00571956],
        [ 0.00571956, -0.05770473,  0.60333333],
    ])


@pytest.fixture
def sorted_():
    return np.array([
        0.6382683737293516, 0.5602684660129408, -0.7922588097422931
    ])


@pytest.fixture
def unsorted_():
    return np.array([-0.79225881,  0.56026847, 0.63826837])


def test_get_tensor_eigenvalues(tensor, sorted_, unsorted_):

    print(get_tensor_eigenvalues(tensor, sort=True))
    print(get_tensor_eigenvalues(tensor, sort=False))
    assert np.all(np.allclose(
        get_tensor_eigenvalues(tensor, sort=True),
        sorted_,
    ))
    assert np.all(np.allclose(
        get_tensor_eigenvalues(tensor, sort=False),
        unsorted_,
    ))

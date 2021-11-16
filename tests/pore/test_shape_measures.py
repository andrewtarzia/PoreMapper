import numpy as np


def test_asphericity(shape_cases):
    test = shape_cases.pore.get_asphericity()
    assert np.isclose(
        test, shape_cases.asphericity, atol=1E-2
    )


def test_relative_shape_anisotropy(shape_cases):
    test = shape_cases.pore.get_relative_shape_anisotropy()
    assert np.isclose(
        test, shape_cases.relative_shape_anisotropy, atol=1E-2
    )

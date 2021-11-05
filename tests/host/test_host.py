import pytest
import os
import numpy as np


def test_host_get_position_matrix(case_data):
    assert np.all(np.allclose(
        case_data.position_matrix1,
        case_data.host.get_position_matrix(),
    ))


def test_host_get_maximum_diameter(case_data):
    test = case_data.host.get_maximum_diameter()
    assert test == case_data.maximum_diameter


def test_host_get_num_atoms(case_data):
    assert case_data.host.get_num_atoms() == case_data.num_atoms


def test_host_get_atoms(case_data):
    for test, atom in zip(case_data.host.get_atoms(), case_data.atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()
        assert test.get_radii() == atom.get_radii()


def test_host_get_centroid(case_data):
    test = case_data.host.get_centroid()
    assert np.all(np.allclose(
        case_data.centroid1,
        test,
    ))


def test_host_with_centroid(case_data):
    test = case_data.host.with_centroid(case_data.centroid2)
    assert np.all(np.allclose(
        case_data.centroid2,
        test.get_centroid(),
    ))

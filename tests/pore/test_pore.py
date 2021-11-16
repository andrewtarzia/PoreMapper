import pytest
import os
import numpy as np


def test_pore_get_position_matrix(case_data):
    test = case_data.pore.get_position_matrix()
    print(test)
    assert np.all(np.allclose(
        case_data.position_matrix,
        test,
    ))


def test_pore_get_num_beads(case_data):
    assert case_data.pore.get_num_beads() == case_data.num_beads


def test_pore_get_maximum_distance_to_com(case_data):
    test = case_data.pore.get_maximum_distance_to_com()
    assert np.isclose(test, case_data.maximum_distance_to_com)


def test_pore_get_mean_distance_to_com(case_data):
    test = case_data.pore.get_mean_distance_to_com()
    assert np.isclose(test, case_data.mean_distance_to_com)


def test_pore_get_volume(case_data):
    test = case_data.pore.get_volume()
    assert np.isclose(test, case_data.volume)


def test_pore_get_intertia_tensor(case_data):
    test = case_data.pore.get_inertia_tensor()
    print(test)
    assert np.all(np.allclose(
        test,
        case_data.inertia_tensor,
    ))

def test_pore_get_asphericity(case_data):
    test = case_data.pore.get_asphericity()
    assert np.isclose(test, case_data.asphericity)

def test_pore_get_relative_shape_anisotropy(case_data):
    test = case_data.pore.get_relative_shape_anisotropy()
    assert np.isclose(test, case_data.relative_shape_anisotropy)

import pytest
import os
import numpy as np


def test_blob_get_position_matrix(case_data):
    assert np.all(np.allclose(
        case_data.position_matrix1,
        case_data.blob.get_position_matrix(),
    ))


def test_blob_get_sigma(case_data):
    assert case_data.blob.get_sigma() == case_data.sigma


def test_blob_get_movable_bead_ids(case_data):
    test = case_data.blob.get_movable_bead_ids()
    assert test == case_data.movable_bead_ids1


def test_blob_with_movable_bead_ids(case_data):
    test = case_data.blob.with_movable_bead_ids(
        case_data.movable_bead_ids2
    ).get_movable_bead_ids()
    assert test == case_data.movable_bead_ids2


def test_blob_get_windows(case_data):
    test_blob = case_data.blob.with_position_matrix(
        case_data.position_matrix2
    )
    test_blob = test_blob.with_movable_bead_ids(
        case_data.movable_bead_ids2,
    )
    test = test_blob.get_windows()
    assert len(case_data.windows) == len(test)
    for i, j in zip(case_data.windows, test):
        assert np.isclose(i, j)


def test_blob_get_num_beads(case_data):
    assert case_data.blob.get_num_beads() == case_data.num_beads


def test_blob_with_centroid(case_data):
    test = case_data.blob.with_centroid(case_data.centroid1)
    assert np.all(np.allclose(
        case_data.centroid1,
        test.get_centroid(),
    ))


def test_blob_get_maximum_diameter(case_data):
    test = case_data.blob.get_maximum_diameter()
    assert np.isclose(test, case_data.maximum_diameter)


def test_blob_with_position_matrix(case_data):
    test = case_data.blob.with_position_matrix(
        case_data.position_matrix2
    )
    assert np.all(np.allclose(
        case_data.position_matrix2,
        test.get_position_matrix(),
    ))



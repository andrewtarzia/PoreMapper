import pytest
import numpy as np
import pore_mapper as pm
from dataclasses import dataclass

@dataclass
class CaseData:
    blob: pm.Blob
    movable_bead_ids1: tuple
    movable_bead_ids2: tuple
    sigma: float
    num_beads: int
    maximum_diameter: float
    position_matrix1: np.ndarray
    position_matrix2: np.ndarray
    centroid1: np.ndarray
    windows: list

@pytest.fixture
def case_data(request):
    sigma = 1.3
    position_matrix1 = np.array([
        [ 0.04358899,  0.,          0.09],
        [-0.05265867,  0.04823966,  0.07],
        [ 0.00757129, -0.08627094,  0.05],
        [ 0.05804137,  0.07570469,  0.03],
        [-0.09797775, -0.01733089,  0.01],
        [ 0.08395259, -0.05340377, -0.01],
        [-0.02476467,  0.09212335, -0.03],
        [-0.03991572, -0.07685529, -0.05],
        [ 0.06708096,  0.02449786, -0.07],
        [-0.04029129,  0.01663166, -0.09],
    ])
    position_matrix2 = position_matrix1 * 10.
    maximum_diameter = 0.311966
    centroid1 = np.array([0., 0., 0.])
    num_beads = 10
    movable_bead_ids1 = tuple(i for i in range(num_beads))
    movable_bead_ids2 = (0, 2, 7)
    windows = [0. for i in movable_bead_ids2]
    return CaseData(
        blob=pm.Blob.init_from_idealised_geometry(
            bead_sigma=1.3,
            num_beads=num_beads,
            sphere_radius=0.1,
        ),
        sigma=sigma,
        movable_bead_ids1=movable_bead_ids1,
        movable_bead_ids2=movable_bead_ids2,
        position_matrix1=position_matrix1,
        position_matrix2=position_matrix2,
        maximum_diameter=maximum_diameter,
        num_beads=num_beads,
        centroid1=centroid1,
        windows=windows,
    )

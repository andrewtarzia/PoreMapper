import pytest
import numpy as np
import pore_mapper as pm
from dataclasses import dataclass

@dataclass
class CaseData:
    pore: pm.Pore
    nonmovable_bead_ids1: tuple
    sigma: float
    num_beads: int
    maximum_distance_to_com: float
    mean_distance_to_com: float
    volume: float
    asphericity: float
    acylindricity: float
    relative_shape_anisotropy: float
    inertia_tensor: np.ndarray
    position_matrix: np.ndarray

@pytest.fixture
def case_data(request):
    sigma = 1.3
    maximum_distance_to_com = 1.1549115
    mean_distance_to_com = 0.960866
    nonmovable_bead_ids1 = (0, 1, 2, 3, 7, 8)
    num_beads = len(nonmovable_bead_ids1)
    blob = pm.Blob.init_from_idealised_geometry(
        bead_sigma=1.3,
        num_beads=10,
        sphere_radius=0.1,
    )
    position_matrix = np.array([
        [ 0.43588989,  0.,          0.9],
        [-0.52658671,  0.48239656,  0.7],
        [ 0.0757129,  -0.86270943,  0.5],
        [ 0.58041368,  0.75704687,  0.3],
        [-0.39915719, -0.76855288, -0.5],
        [ 0.67080958,  0.24497858, -0.7],
    ])
    blob = blob.with_position_matrix(blob.get_position_matrix()*10)
    volume = 10.439176
    asphericity = 0.2321977
    acylindricity = 0.09692476
    relative_shape_anisotropy = 0.01524040
    inertia_tensor = np.array([
        [ 0.76346367, -0.09852765,  0.00571956],
        [-0.09852765,  0.633203,   -0.05770473],
        [ 0.00571956, -0.05770473,  0.60333333],
    ])
    return CaseData(
        pore=pm.Pore(blob, nonmovable_bead_ids1),
        sigma=sigma,
        volume=volume,
        nonmovable_bead_ids1=nonmovable_bead_ids1,
        position_matrix=position_matrix,
        maximum_distance_to_com=maximum_distance_to_com,
        mean_distance_to_com=mean_distance_to_com,
        num_beads=num_beads,
        asphericity=asphericity,
        acylindricity=acylindricity,
        relative_shape_anisotropy=relative_shape_anisotropy,
        inertia_tensor=inertia_tensor,
    )

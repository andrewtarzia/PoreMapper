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
        relative_shape_anisotropy=relative_shape_anisotropy,
        inertia_tensor=inertia_tensor,
    )


@dataclass
class ShapeCase:
    pore: pm.Pore
    asphericity: float
    relative_shape_anisotropy: float


@pytest.fixture(
    params=(
        ShapeCase(
            pore=pm.Pore(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=10,
                    sphere_radius=0.1,
                ),
                nonmovable_bead_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            ),
            asphericity=0.0,
            relative_shape_anisotropy=0.0,
        ),
        # Cube.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,     1.0000],
                    [1.0000,   -1.0000,     1.0000],
                    [-1.0000,   -1.0000,     1.0000],
                    [-1.0000,    1.0000,     1.0000],
                    [1.0000,    1.0000,    -1.0000],
                    [1.0000,   -1.0000,    -1.0000],
                    [-1.0000,   -1.0000,    -1.0000],
                    [-1.0000,    1.0000,    -1.0000],
                ]),
            ),
            asphericity=0.0,
            relative_shape_anisotropy=0.0,
        ),
        # Larger cube.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [50.000,    50.000,    50.000],
                    [50.000,   -50.000,    50.000],
                    [-50.000,   -50.000,    50.000],
                    [-50.000,    50.000,    50.000],
                    [50.000,    50.000,   -50.000],
                    [50.000,   -50.000,   -50.000],
                    [-50.000,   -50.000,   -50.000],
                    [-50.000,    50.000,   -50.000],
                ]),
            ),
            asphericity=0.0,
            relative_shape_anisotropy=0.0,
        ),
        # Rectangle.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,    1.0000],
                    [1.0000,   -1.0000,    1.0000],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,    1.0000,   -1.5000],
                    [1.0000,   -1.0000,   -1.5000],
                    [-1.0000,   -1.0000,   -1.5000],
                    [-1.0000,    1.0000,   -1.5000],
                ]),
            ),
            asphericity=0.3125,
            relative_shape_anisotropy=0.0074,
        ),
        # Rectangle.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,    1.0000],
                    [1.0000,   -1.0000,    1.0000],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,    1.0000,   -2.0000],
                    [1.0000,   -1.0000,   -2.0000],
                    [-1.0000,   -1.0000,   -2.0000],
                    [-1.0000,    1.0000,   -2.0000],
                ]),
            ),
            asphericity=0.75,
            relative_shape_anisotropy=0.0277,
        ),
        # Rectangle.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,    1.0000],
                    [1.0000,   -1.0000,    1.0000],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,    1.0000,   -3.0000],
                    [1.0000,   -1.0000,   -3.0000],
                    [-1.0000,   -1.0000,   -3.0000],
                    [-1.0000,    1.0000,   -3.0000],
                ]),
            ),
            asphericity=2.0,
            relative_shape_anisotropy=0.0816,
        ),
        # Rectangle.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,    1.0000],
                    [1.0000,   -1.0000,    1.0000],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,    1.0000,   -4.0000],
                    [1.0000,   -1.0000,   -4.0000],
                    [-1.0000,   -1.0000,   -4.0000],
                    [-1.0000,    1.0000,   -4.0000],
                ]),
            ),
            asphericity=3.75,
            relative_shape_anisotropy=0.12755,
        ),
        # Rectangle.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.0000,    1.0000],
                    [1.0000,   -1.0000,    1.0000],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,    1.0000,   -20.0000],
                    [1.0000,   -1.0000,   -20.0000],
                    [-1.0000,   -1.0000,   -20.0000],
                    [-1.0000,    1.0000,   -20.0000],
                ]),
            ),
            asphericity=99.75,
            relative_shape_anisotropy=0.2496,
        ),
        # mess.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    1.8474,    1.9756],
                    [1.0000,   -0.3938,    0.4837],
                    [-1.0000,   -1.0000,    1.0000],
                    [-1.0000,    0.7611,    1.9942],
                    [1.0000,    1.1679,   -1.2583],
                    [1.0000,   -1.0193,   -1.5293],
                    [-1.0000,   -1.0000,   -1.0000],
                    [-1.0000,    1.0000,   -1.0000],
                ]),
            ),
            asphericity=1.019,
            relative_shape_anisotropy=0.02496,
        ),
        # mess.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    2.0695,    2.6300],
                    [1.0000,   -2.0268,    1.9880],
                    [-1.0000,   -0.4682,    2.6646],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,   -0.2578,    0.1602],
                    [1.0000,   -1.3950,   -1.2936],
                    [-1.0000,   -0.6837,   -2.7129],
                    [-1.0000,    1.0000,   -1.0000],
                ]),
            ),
            asphericity=2.0029,
            relative_shape_anisotropy=0.04574,
        ),
        # Larger, but same mess.
        ShapeCase(
            pore=pm.Pore.init(
                blob=pm.Blob.init_from_idealised_geometry(
                    bead_sigma=1.,
                    num_beads=8,
                    sphere_radius=0.1,
                ),
                num_beads=8,
                sigma=1.,
                beads=(pm.Bead(i, 1.) for i in range(8)),
                position_matrix=np.array([
                    [1.0000,    2.0695,    2.6300],
                    [1.0000,   -2.0268,    1.9880],
                    [-1.0000,   -0.4682,    2.6646],
                    [-1.0000,    1.0000,    1.0000],
                    [1.0000,   -0.2578,    0.1602],
                    [1.0000,   -1.3950,   -1.2936],
                    [-1.0000,   -0.6837,   -2.7129],
                    [-1.0000,    1.0000,   -1.0000],
                ])*30,
            ),
            asphericity=1802.611,
            relative_shape_anisotropy=0.04574,
        ),
    )
)
def shape_cases(request):
    return request.param
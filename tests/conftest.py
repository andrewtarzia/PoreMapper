import pytest
import numpy as np
import roll_gather as rg


@pytest.fixture(
    params=(
        (rg.Atom(id=0, element_string='N'), 0, 'N', 1.4882656711484),
        (rg.Atom(id=65, element_string='P'), 65, 'P', 1.925513788),
        (rg.Atom(id=2, element_string='C'), 2, 'C', 1.60775485914852),
    )
)
def atom_info(request):
    return request.param


@pytest.fixture(
    params=(
        (rg.Bead(id=0, sigma=1.), 0, 1.),
        (rg.Bead(id=65, sigma=2.2), 65, 2.2),
        (rg.Bead(id=2, sigma=1.4), 2, 1.4),
    )
)
def bead_info(request):
    return request.param


@pytest.fixture
def atoms():
    return [rg.Atom(0, 'C'), rg.Atom(1, 'C')]


@pytest.fixture
def position_matrix():
    return np.array([[0, 0, 0], [0, 1.5, 0]])


@pytest.fixture
def position_matrix2():
    return np.array([[0, 0, 0], [0, 3, 0]])


@pytest.fixture
def displacement():
    return np.array([0, 1, 0])


@pytest.fixture
def displaced_position_matrix():
    return np.array([[0, 1, 0], [0, 2.5, 0]])


@pytest.fixture
def centroid():
    return np.array([0, 0.75, 0])


@pytest.fixture
def num_atoms():
    return 2


@pytest.fixture
def Host(atoms, position_matrix):
    return rg.Host(
        atoms=atoms,
        position_matrix=position_matrix
    )


@pytest.fixture
def final_pos_mat():
    return np.array([
        [0.29028551,  1.12756372, -1.21825898],
        [1.23559041,  2.11685589,  0.89545513]
    ])


@pytest.fixture
def spinner():
    return rg.Inflater(bead_sigma=1.0)

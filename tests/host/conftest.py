import pytest
import numpy as np
import roll_gather as rg
from dataclasses import dataclass

@dataclass
class CaseData:
    host: rg.Host
    atoms: tuple
    num_atoms: int
    maximum_diameter: int
    position_matrix1: np.ndarray
    centroid1: np.ndarray
    centroid2: np.ndarray

@pytest.fixture
def case_data(request):
    atoms = (rg.Atom(0, 'C'), rg.Atom(1, 'C'))
    position_matrix1 = np.array([[0, 0, 0], [0, 1.5, 0]])
    maximum_diameter = 1.5
    centroid1 = np.array([0., 0.75, 0.])
    centroid2 = np.array([0., 0., 0.])
    num_atoms = 2
    return CaseData(
        host=rg.Host(
            atoms=atoms,
          position_matrix=position_matrix1,
        ),
        position_matrix1=position_matrix1,
        maximum_diameter=maximum_diameter,
        num_atoms=num_atoms,
        atoms=atoms,
        centroid1=centroid1,
        centroid2=centroid2,
    )




@pytest.fixture
def host(atoms, position_matrix):
    return rg.Host(
        atoms=atoms,
        position_matrix=position_matrix,
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

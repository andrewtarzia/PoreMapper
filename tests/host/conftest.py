import pytest
import numpy as np
import pore_mapper as pm
from dataclasses import dataclass

@dataclass
class CaseData:
    host: pm.Host
    atoms: tuple
    num_atoms: int
    maximum_diameter: float
    position_matrix1: np.ndarray
    centroid1: np.ndarray
    centroid2: np.ndarray

@pytest.fixture
def case_data(request):
    atoms = (pm.Atom(0, 'C'), pm.Atom(1, 'C'))
    position_matrix1 = np.array([[0, 0, 0], [0, 1.5, 0]])
    maximum_diameter = 1.5
    centroid1 = np.array([0., 0.75, 0.])
    centroid2 = np.array([0., 0., 0.])
    num_atoms = 2
    return CaseData(
        host=pm.Host(
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

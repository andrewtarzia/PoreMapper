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

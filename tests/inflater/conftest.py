from dataclasses import dataclass

import numpy as np
import pore_mapper as pm
import pytest
import stk

_cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        building_blocks=(
            stk.BuildingBlock(
                smiles="NCCN",
                functional_groups=[stk.PrimaryAminoFactory()],
            ),
            stk.BuildingBlock(
                smiles="O=CC(C=O)C=O",
                functional_groups=[stk.AldehydeFactory()],
            ),
        ),
        optimizer=stk.MCHammer(),
    ),
)
_host = pm.Host(
    atoms=(
        pm.Atom(id=i.get_id(), element_string=i.__class__.__name__)
        for i in _cage.get_atoms()
    ),
    position_matrix=_cage.get_position_matrix(),
)


@dataclass
class CageCase:
    host: pm.Host
    centroid: np.ndarray
    sigma: float
    pore_volume: float
    name: str


@pytest.fixture(
    scope="session",
    params=(
        lambda name: CageCase(
            host=_host,
            centroid=_cage.get_centroid(),
            sigma=1.0,
            pore_volume=252.43952526926182,
            name=name,
        ),
        lambda name: CageCase(
            host=_host,
            centroid=_cage.get_centroid(),
            sigma=0.5,
            pore_volume=377.8770081159259,
            name=name,
        ),
        lambda name: CageCase(
            host=_host,
            centroid=_cage.get_centroid(),
            sigma=2.0,
            pore_volume=5.901644478091131,
            name=name,
        ),
        lambda name: CageCase(
            host=_host,
            centroid=_cage.get_centroid() + np.array((0, 3, 0)),
            sigma=1.0,
            pore_volume=0.0073443130418483155,
            name=name,
        ),
        lambda name: CageCase(
            host=_host,
            centroid=np.array((0, 0, 0)),
            sigma=1.0,
            pore_volume=217.2292473830309,
            name=name,
        ),
    ),
)
def cage_case(request) -> CageCase:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )

import numpy as np
import pore_mapper as pm


def test_inflater(cage_case):
    calculator = pm.Inflater(
        bead_sigma=cage_case.sigma,
        centroid=cage_case.centroid,
    )
    final_result = calculator.get_inflated_blob(host=cage_case.host)
    pore = final_result.pore
    print(pore.get_volume())
    print(cage_case.host.get_centroid(), cage_case.centroid)
    # cage_case.host.write_xyz_file(f"./{cage_case.name}_final.xyz")
    # pore.write_xyz_file(f"./{cage_case.name}_pore_final.xyz")
    assert np.isclose(
        pore.get_volume(), cage_case.pore_volume, atol=1e-6, rtol=0
    )

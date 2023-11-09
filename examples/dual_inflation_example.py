import os

import numpy as np
import pore_mapper as pm
import stk


def run_calculation(
    prefix,
    stk_molecule,
    centroid1,
    centroid2,
    pore_distance1,
    pore_distance2,
):
    # Read in host from xyz file.
    host = pm.Host(
        atoms=(
            pm.Atom(id=i.get_id(), element_string=i.__class__.__name__)
            for i in stk_molecule.get_atoms()
        ),
        position_matrix=stk_molecule.get_position_matrix(),
    )

    # Number of centroids can be any number:
    pore_distances = []
    for i, centroid in enumerate((centroid1, centroid2)):
        # Define calculator object.
        calculator = pm.Inflater(bead_sigma=1.2, centroid=centroid)

        # Run calculator on host object, analysing output.
        final_result = calculator.get_inflated_blob(host=host)
        pore = final_result.pore
        blob = final_result.pore.get_blob()
        print(
            f"{prefix} - centroid {i}: "
            f"mean_distance: {pore.get_mean_distance_to_com()}"
        )
        pore_distances.append(pore.get_mean_distance_to_com())

        # Do final structure.
        host.write_xyz_file(f"example_output/{prefix}_{i}_final.xyz")
        blob.write_xyz_file(f"example_output/{prefix}_{i}_blob_final.xyz")
        pore.write_xyz_file(f"example_output/{prefix}_{i}_pore_final.xyz")

    # A quick check for no changes.
    print(pore_distances)
    assert np.isclose(pore_distances[0], pore_distance1, atol=1e-6, rtol=0)
    assert np.isclose(pore_distances[1], pore_distance2, atol=1e-6, rtol=0)


def main():
    if not os.path.exists("example_output"):
        os.mkdir("example_output")

    tests = {
        "Cage_G_17_0_0_aa": {
            "centroid1": np.array((0, 0, 0)),
            "centroid2": np.array((-0.11233751, 4.14297331, 6.70839747)),
            "pore_distance1": 1.8961487347022379,
            "pore_distance2": 0.09999999998683182,
        },
        "Cage_G_17_63_6_aa": {
            "centroid1": np.array((0.41359275, -4.41492435, 12.58085734)),
            "centroid2": np.array((0, 0, 0)),
            "pore_distance1": 2.3039183754161665,
            "pore_distance2": 2.3069917836544382,
        },
    }

    for prefix in tests:
        run_calculation(
            prefix=prefix,
            stk_molecule=stk.BuildingBlock.init_from_file(f"{prefix}.mol"),
            centroid1=tests[prefix]["centroid1"],
            centroid2=tests[prefix]["centroid2"],
            pore_distance1=tests[prefix]["pore_distance1"],
            pore_distance2=tests[prefix]["pore_distance2"],
        )


if __name__ == "__main__":
    main()

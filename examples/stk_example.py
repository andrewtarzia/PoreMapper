import os

import numpy as np
import pore_mapper as pm
import stk


def main():
    if not os.path.exists("example_output"):
        os.mkdir("example_output")

    cage = stk.ConstructedMolecule(
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
    host = pm.Host(
        atoms=(
            pm.Atom(id=i.get_id(), element_string=i.__class__.__name__)
            for i in cage.get_atoms()
        ),
        position_matrix=cage.get_position_matrix(),
    )

    prefix = "stk"

    # Define calculator object.
    calculator = pm.Inflater(bead_sigma=1.2, centroid=host.get_centroid())

    # Run calculator on host object, analysing output.
    print(f"doing {prefix}")
    final_result = calculator.get_inflated_blob(host=host)
    pore = final_result.pore
    blob = final_result.pore.get_blob()
    windows = pore.get_windows()
    print(final_result)
    print(
        f"step: {final_result.step}\n"
        f"num_movable_beads: {final_result.num_movable_beads}\n"
        f"windows: {windows}\n"
        f"blob: {blob}\n"
        f"pore: {pore}\n"
        f"blob_max_diam: {blob.get_maximum_diameter()}\n"
        f"pore_max_rad: {pore.get_maximum_distance_to_com()}\n"
        f"pore_mean_rad: {pore.get_mean_distance_to_com()}\n"
        f"pore_volume: {pore.get_volume()}\n"
        f"num_windows: {len(windows)}\n"
    )
    print()

    # Do final structure.
    host.write_xyz_file(f"example_output/{prefix}_final.xyz")
    blob.write_xyz_file(f"example_output/{prefix}_blob_final.xyz")
    pore.write_xyz_file(f"example_output/{prefix}_pore_final.xyz")

    # A quick check for no changes.
    assert np.isclose(pore.get_volume(), 148.89022549492185, atol=1e-6, rtol=0)


if __name__ == "__main__":
    main()

import os
import roll_gather as rg


def run_calculation(prefix):
    # Read in host from xyz file.
    host = rg.Host.init_from_xyz_file(path=f'{prefix}.xyz')
    host = host.with_centroid([0., 0., 0.])

    # Define calculator object.
    calculator = rg.Inflater(bead_sigma=1.0)

    # Run calculator on host object, analysing output.
    print(f'doing {prefix}')
    final_result = calculator.get_inflated_blob(host=host)
    pore = final_result.pore
    blob = final_result.pore.get_blob()
    windows = pore.get_windows()
    print(final_result)
    print(
        f'step: {final_result.step}\n'
        f'num_movable_beads: {final_result.num_movable_beads}\n'
        f'windows: {windows}\n'
        f'blob: {blob}\n'
        f'pore: {pore}\n'
        f'blob_max_diam: {blob.get_maximum_diameter()}\n'
        f'pore_max_rad: {pore.get_maximum_distance_to_com()}\n'
        f'pore_mean_rad: {pore.get_mean_distance_to_com()}\n'
        f'pore_volume: {pore.get_volume()}\n'
        f'num_windows: {len(windows)}\n'
        f'max_window_size: {max(windows)}\n'
        f'min_window_size: {min(windows)}\n'
        f'asphericity: {pore.get_asphericity()}\n'
        f'acylindricity: {pore.get_acylindricity()}\n'
        f'shape anisotropy: {pore.get_relative_shape_anisotropy()}\n'
    )
    print()

    # Do final structure.
    host.write_xyz_file(f'min_example_output/{prefix}_final.xyz')
    blob.write_xyz_file(f'min_example_output/{prefix}_blob_final.xyz')
    pore.write_xyz_file(f'min_example_output/{prefix}_pore_final.xyz')


def main():

    if not os.path.exists('min_example_output'):
        os.mkdir('min_example_output')

    names = (
        'cc3', 'moc2', 'moc1', 'hogrih_cage',
        'hogsoo_cage', 'hogsii_cage', 'yamashina_cage_'
    )

    for prefix in names:
        run_calculation(prefix)

if __name__ == '__main__':
    main()

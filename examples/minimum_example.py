import os
from numpy.lib.function_base import average
import roll_gather as rg
import matplotlib.pyplot as plt
import pywindow as pw
import time


def run_pywindow(prefix):
    molsys = pw.MolecularSystem.load_file(f'{prefix}.xyz')
    mol = molsys.system_to_molecule()
    mol.full_analysis()

    return mol.properties


def run_calculation(prefix):
    # Read in host from xyz file.
    host = rg.Host.init_from_xyz_file(path=f'{prefix}.xyz')
    host = host.with_centroid([0., 0., 0.])

    # Define calculator object.
    calculator = rg.Inflater(bead_sigma=1.0)

    # Run calculator on host object, analysing output.
    blob_properties = {}
    stime = time.time()
    stime2 = time.time()
    for step_result in calculator.inflate_blob(host=host):
        print(f'step time: {time.time() - stime2}')
        print(step_result)
        print(
            f'step: {step_result.step}, '
            f'num_movable_beads: {step_result.num_movable_beads}'
        )
        pore = step_result.pore
        blob = step_result.pore.get_blob()
        if step_result.step % 10 == 0:
            blob.write_xyz_file(
                f'min_example_output/'
                f'{prefix}_blob_{step_result.step}.xyz'
            )
            pore.write_xyz_file(
                f'min_example_output/'
                f'{prefix}_pore_{step_result.step}.xyz'
            )

        windows = pore.get_windows()
        print(f'windows: {windows}\n')
        blob_properties[step_result.step] = {
            'num_movable_beads': (
                step_result.num_movable_beads/blob.get_num_beads()
            ),
            'blob_max_diam': blob.get_maximum_diameter(),
            'pore_max_rad': pore.get_maximum_distance_to_com(),
            'pore_mean_rad': pore.get_mean_distance_to_com(),
            'pore_volume': pore.get_volume(),
            'pore_volume': pore.get_volume(),
            'num_windows': len(windows),
            'max_window_size': max(windows),
            'avg_window_size': average(windows),
            'min_window_size': min(windows),
            'asphericity': pore.get_asphericity(),
            'acylindricity': pore.get_acylindricity(),
            'shape_anisotropy': pore.get_relative_shape_anisotropy(),
        }
        stime2 = time.time()

    print(f'run time: {time.time() - stime}')
    # Do final structure.
    blob.write_xyz_file(
        f'min_example_output/{prefix}_blob_{step_result.step}.xyz'
    )
    pore.write_xyz_file(
        f'min_example_output/{prefix}_pore_{step_result.step}.xyz'
    )

    return blob_properties


def plot(properties, pywindow, filename):
    pw_num_windows = len(pywindow['windows']['diameters'])
    pw_pore_volume = pywindow['pore_volume_opt']
    try:
        pw_min_window_size = min(pywindow['windows']['diameters'])/2
        pw_avg_window_size = average(
            pywindow['windows']['diameters']
        )/2
        pw_max_window_size = max(pywindow['windows']['diameters'])/2
    except ValueError:
        pw_min_window_size = 0
        pw_avg_window_size = 0
        pw_max_window_size = 0

    pw_pore_diameter = pywindow['pore_diameter_opt']['diameter']

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
    ax[0].plot(
        [i for i in properties],
        [properties[i]['num_movable_beads'] for i in properties],
        c='k',
        lw=2,
        label='frac. movable beads',
    )
    ax[0].plot(
        [i for i in properties],
        [properties[i]['num_windows'] for i in properties],
        c='green',
        lw=2,
        label='num. windows',
    )
    ax[0].axhline(
        pw_num_windows, c='r', lw=2, label='pw: num. windows.'
    )
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_ylabel('value', fontsize=16)
    ax[0].legend(fontsize=16)

    ax[1].plot(
        [i for i in properties],
        [properties[i]['pore_volume'] for i in properties],
        c='#648FFF',
        lw=2,
    )
    ax[1].axhline(
        pw_pore_volume, c='r', lw=2, label='pw'
    )
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    ax[1].set_ylabel(r'pore vol. [$\mathrm{\AA}^3$]', fontsize=16)
    ax[1].legend(fontsize=16)

    ax[2].plot(
        [i for i in properties],
        [properties[i]['max_window_size'] for i in properties],
        c='k',
        lw=2,
        linestyle='--',
        label='max',
    )
    ax[2].plot(
        [i for i in properties],
        [properties[i]['avg_window_size'] for i in properties],
        c='r',
        lw=2,
        linestyle='--',
        label='avg',
    )
    ax[2].plot(
        [i for i in properties],
        [properties[i]['min_window_size'] for i in properties],
        c='b',
        lw=2,
        linestyle='--',
        label='min',
    )
    ax[2].axhline(
        pw_min_window_size, c='k', lw=2, label='pw: min'
    )
    ax[2].axhline(
        pw_avg_window_size, c='r', lw=2, label='pw: avg'
    )
    ax[2].axhline(
        pw_max_window_size, c='b', lw=2, label='pw: max'
    )
    ax[2].tick_params(axis='both', which='major', labelsize=16)
    ax[2].set_ylabel(r'window rad. [$\mathrm{\AA}$]', fontsize=16)
    ax[2].legend(fontsize=16, ncol=3)

    ax[3].plot(
        [i for i in properties],
        [properties[i]['asphericity'] for i in properties],
        c='k',
        lw=2,
        label='asphericity',
    )
    ax[3].plot(
        [i for i in properties],
        [properties[i]['shape_anisotropy'] for i in properties],
        c='b',
        lw=2,
        label='rel. shape anisotropy',
    )
    ax[3].plot(
        [i for i in properties],
        [properties[i]['acylindricity'] for i in properties],
        c='r',
        lw=2,
        label='acylindricity',
    )
    ax[3].tick_params(axis='both', which='major', labelsize=16)
    ax[3].set_ylabel('value', fontsize=16)
    ax[3].legend(fontsize=16)

    # ax[-1].plot(
    #     [i for i in blob_properties],
    #     [blob_properties[i]['blob_max_diam'] for i in blob_properties],
    #     c='#DC267F',
    #     label='blob',
    #     lw=2,
    # )
    ax[-1].plot(
        [i for i in properties],
        [properties[i]['pore_max_rad']*2 for i in properties],
        c='#785EF0',
        label='max',
        lw=2,
    )
    ax[-1].plot(
        [i for i in properties],
        [properties[i]['pore_mean_rad']*2 for i in properties],
        c='#648FFF',
        label='mean',
        lw=2,
        linestyle='--'
    )
    ax[-1].axhline(
        pw_pore_diameter, c='r', lw=2, label='pw: diameter'
    )
    ax[-1].tick_params(axis='both', which='major', labelsize=16)
    ax[-1].set_xlim(0, None)
    ax[-1].set_xlabel('step', fontsize=16)
    ax[-1].set_ylabel(r'pore diam. [$\mathrm{\AA}$]', fontsize=16)
    ax[-1].legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=360,
        bbox_inches='tight'
    )



def main():

    if not os.path.exists('min_example_output'):
        os.mkdir('min_example_output')

    names = ('cc3', )

    for prefix in names:
        pywindow_properties = run_pywindow(prefix)
        blob_properties = run_calculation(prefix)
        # Do a plot of properties.
        plot(
            properties=blob_properties,
            pywindow=pywindow_properties,
            filename=f'inflation_example_{prefix}.pdf',
        )

if __name__ == '__main__':
    main()

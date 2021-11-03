import os
import roll_gather as rg

import matplotlib.pyplot as plt

if not os.path.exists('min_example_output'):
    os.mkdir('min_example_output')

# Read in host from xyz file.
host = rg.Host.init_from_xyz_file(path='cc3.xyz')
print(host)

# Define calculator object.
calculator = rg.Inflater(
    step_size=0.1,
    bead_sigma=0.5,
    num_beads=100,
    num_steps=100,
)

# Run calculator on host object, analysing output.
blob_properties = {}
for step_result in calculator.inflate_blob(host=host):
    print(step_result)
    print(
        f'step: {step_result.step}, '
        f'num_movable_beads: {step_result.num_movable_beads}'
    )
    blob = step_result.blob
    print(blob)
    pore = step_result.pore
    print(pore)
    if step_result.step % 5 == 0:
        blob.write_xyz_file(
            f'min_example_output/blob_{step_result.step}.xyz'
        )
        pore.write_xyz_file(
            f'min_example_output/pore_{step_result.step}.xyz'
        )
    blob_properties[step_result.step] = {
        'num_movable_beads': step_result.num_movable_beads,
        'blob_max_diam': step_result.blob.get_maximum_diameter(),
        'pore_max_diam': step_result.pore.get_maximum_distance_to_com(),
        'pore_mean_diam': step_result.pore.get_mean_distance_to_com(),
    }

# Do final structure.
blob.write_xyz_file(
    f'min_example_output/blob_{step_result.step}.xyz'
)
pore.write_xyz_file(
    f'min_example_output/pore_{step_result.step}.xyz'
)

# Do a plot of properties.
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
ax[0].plot(
    [i for i in blob_properties],
    [
        blob_properties[i]['num_movable_beads'] for i in blob_properties
    ],
    c='k',
    lw=2,
)
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[0].set_ylabel('num. movable beads', fontsize=16)

ax[1].plot(
    [i for i in blob_properties],
    [blob_properties[i]['blob_max_diam'] for i in blob_properties],
    c='#DC267F',
    label='blob',
    lw=2,
)
ax[1].plot(
    [i for i in blob_properties],
    [blob_properties[i]['pore_max_diam'] for i in blob_properties],
    c='#785EF0',
    label='pore max',
    lw=2,
)
ax[1].plot(
    [i for i in blob_properties],
    [blob_properties[i]['pore_mean_diam'] for i in blob_properties],
    c='#648FFF',
    label='pore mean',
    lw=2,
    linestyle='--'
)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[1].set_xlim(0, None)
ax[1].set_xlabel('step', fontsize=16)
ax[1].set_ylabel(r'max. diam. [$\mathrm{\AA}$]', fontsize=16)
ax[1].legend(fontsize=16)
fig.tight_layout()
fig.savefig(
    'inflation_example.pdf',
    dpi=360,
    bbox_inches='tight'
)

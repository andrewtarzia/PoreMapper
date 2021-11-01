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
    rotation_step_size=1,
    bead_sigma=1.5,
    max_size_modifier=1,
    max_beads=100,
    num_dynamics_steps=100,
    nonbond_epsilon=1,
    beta=3,
    random_seed=1000,
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
    blob.write_xyz_file(
        f'min_example_output/blob_{step_result.step}.xyz'
    )
    pore.write_xyz_file(
        f'min_example_output/pore_{step_result.step}.xyz'
    )
    blob_properties[step_result.step] = {
        'num_movable_beads': step_result.num_movable_beads,
        'max_diam': step_result.blob.get_maximum_diameter(),
    }

# Do a plot of properties.
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
ax[0].plot(
    [i for i in blob_properties],
    [
        blob_properties[i]['num_movable_beads'] for i in blob_properties
    ],
)
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[0].set_ylabel('num_movable_beads', fontsize=16)

ax[1].plot(
    [i for i in blob_properties],
    [blob_properties[i]['max_diam'] for i in blob_properties],
)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[1].set_xlabel('step', fontsize=16)
ax[1].set_ylabel('max. diameter [AA]', fontsize=16)
fig.tight_layout()
fig.savefig(
    'inflation_example.pdf',
    dpi=360,
    bbox_inches='tight'
)

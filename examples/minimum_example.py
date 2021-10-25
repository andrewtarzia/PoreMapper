import os
import roll_gather as rg

if not os.path.exists('min_example_output'):
    os.mkdir('min_example_output')

# Read in host from xyz file.
host = rg.Host.init_from_xyz_file(path='moc2.xyz')
print(host)

# Define calculator object.
calculator = rg.Roller(
    step_size=1.,
    rotation_step_size=1,
    bead_sigma=1.5,
    max_size_modifier=1,
    max_beads=60,
    num_dynamics_steps=4000,
    nonbond_epsilon=1,
    beta=3,
    random_seed=1000,
)

# Run calculator on host object, analysing output.
blob_properties = {}
for step_result in calculator.mould_blob(host=host):
    print(step_result)
    print(
        f'step: {step_result.step}, '
        f'potential: {step_result.potential}'
    )
    blob = step_result.blob
    print(blob)
    blob.write_xyz_file(
        f'min_example_output/blob_{step_result.step}.xyz'
    )
    if step_result.step % 50 == 0:
        blob.write_properties(
            path=(
                'min_example_output/'
                f'blob_{step_result.step}.json'
            ),
            potential=step_result.potential,
        )

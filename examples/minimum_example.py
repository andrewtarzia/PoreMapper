import roll_gather as rg

# Read in host from xyz file.
host = rg.Host.init_from_xyz_file(path='cc3.xyz')
print(host)

# Define calculator object.
calculator = rg.Roller(
    step_size=0.5,
    rotation_step_size=5,
    bead_sigma=1.2,
    max_beads=10,
    num_steps=10,
    nonbond_epsilon=5,
    beta=2,
    random_seed=1000,
)

# Run calculator on host object, analysing output.
blob_properties = {}
for step_result in calculator.grow_blob(host=host):
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
    blob_properties[step_result.step] = blob.get_properties()

rg.write_blob_properties_over_time(
    properties_dict=blob_properties,
    output_file='min_example_output/blob_properties.csv,'
)

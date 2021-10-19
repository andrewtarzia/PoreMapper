import roll_gather as rg

# Read in host from xyz file.
host = rg.Host.init_from_xyz_file('cc3.xyz')

# Define calculator object.
calculator = rg.Roller(
    step_size=0.5,
    rotation_step_size=5,
    bead_type=rg.Bead(sigma=1.2),
    max_beads=10,
    num_steps=10,
)

# Run calculator on host object, analysing output.
blob_properties = {}
for step_result in calculator.grow_blob(host=host):
    print(step_result)
    print(step_result.get_step(), step_result.get_potential())
    blob = step_result.get_blob()
    print(blob)
    blob.write_xyz_file(
        f'min_example_output/blob_{step_result.get_step()}.xyz'
    )
    blob_properties[step_result.get_step()] = blob.get_properties()

rg.write_blob_properties_over_time(
    properties_dict=blob_properties,
    output_file='min_example_output/blob_properties.csv,'
)

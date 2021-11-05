import numpy as np


def test_check_steric(spinner, nonbond_potentials):
    for i, d in enumerate([1, 1.2, 1.4, 1.6, 1.8, 2.0, 4]):
        test = spinner._nonbond_potential(distance=d)
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)


def test_opt(spinner, smolecule, final_pos_mat, final_potential):
    test = spinner.get_final_conformer(smolecule)
    assert np.all(np.allclose(
        final_pos_mat,
        test.get_position_matrix(),
    ))
    assert np.isclose(
        spinner._compute_potential(test),
        final_potential
    )


def test_opt_test_move(spinner):
    # Do not test random component.
    assert spinner._test_move(curr_pot=-1, new_pot=-2)

"""
Roller
======

#. :class:`.Roller`

Generator of blob guests using nonbonded interactions and growth.

"""

from __future__ import annotations
from collections import abc
import typing

import numpy as np

from itertools import combinations
from scipy.spatial.distance import cdist
import random

# from .supramolecule import SupraMolecule
from .bead import Bead
from .host import Host
from .blob import Blob
from .step_result import StepResult
from .utilities import rotation_matrix_arbitrary_axis


class Roller:
    """
    Grow guest blob.

    """

    def __init__(
        self,
        step_size: float,
        rotation_step_size: float,
        bead_type: Bead,
        max_beads: int,
        num_steps: int,
        nonbond_epsilon: typing.Optional[float] = 5.,
        beta: typing.Optional[float] = 2.,
        random_seed: typing.Optional[int] = 1000,
    ):
        """
        Initialize a :class:`Spinner` instance.

        Parameters:

            step_size:
                The relative size of the step to take during step.

            rotation_step_size:
                The relative size of the rotation to take during step.

            bead_type:
                Bead to use in Blob.

            max_beads:
                Maximum number of beads in Blob.

            num_steps:
                Number of steps to run growth for.

            nonbond_epsilon:
                Value of epsilon used in the nonbond potential in MC
                moves. Determines strength of the nonbond potential.

            beta:
                Value of beta used in the in MC moves. Beta takes the
                place of the inverse boltzmann temperature.

            random_seed:
                Random seed to use for MC algorithm. Should only be set
                to ``None`` if system-based random seed is desired.

        """

        self._step_size = step_size
        self._rotation_step_size = rotation_step_size
        self._bead_type = bead_type
        self._max_beads = max_beads
        self._num_steps = num_steps
        self._nonbond_epsilon = nonbond_epsilon
        self._beta = beta
        if random_seed is None:
            np.random.seed()
            random.seed()
        else:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def _nonbond_potential(
        self,
        distance: np.ndarray,
        sigma: float,
    ) -> float:
        """
        Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (sigma/distance) ** 12 - (sigma/distance) ** 6
            )
        )

    def _compute_potential(self, host: Host, blob: Blob) -> float:
        position_matrices = (
            host.get_position_matrix(),
            blob.get_position_matrix(),
        )
        nonbonded_potential = 0
        for pos_mat_pair in combinations(position_matrices, 2):
            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigma=blob.get_sigma(),
                )
            )

        return nonbonded_potential

    def _translate_atoms_along_vector(self, mol, vector):
        return mol.with_displacement(vector)

    def _rotate_atoms_by_angle(self, mol, angle, axis, origin):
        new_position_matrix = mol.get_position_matrix()
        # Set the origin of the rotation to "origin".
        new_position_matrix = new_position_matrix - origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(angle, axis)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        new_position_matrix = (rot_mat @ new_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        new_position_matrix = new_position_matrix + origin

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def _test_move(self, curr_pot, new_pot):

        if new_pot < curr_pot:
            return True
        else:
            exp_term = np.exp(-self._beta*(new_pot-curr_pot))
            rand_number = random.random()

            if exp_term > rand_number:
                return True
            else:
                return False

    def _run_step(self, supramolecule):

        component_list = list(supramolecule.get_components())
        component_sizes = {
            i: mol.get_num_atoms()
            for i, mol in enumerate(component_list)
        }
        max_size = max(component_sizes.values())
        # Select a guest randomly to move and reorient.
        # Do not move or rotate largest component if same size.
        if len(set(component_sizes.values())) > 1:
            targ_comp_id = random.choice([
                i for i in range(len(component_list))
                if component_sizes[i] != max_size
            ])
        else:
            targ_comp_id = random.choice([
                i for i in range(len(component_list))
            ])

        targ_comp = component_list[targ_comp_id]

        # Random number from -1 to 1 for multiplying translation.
        rand = (random.random() - 0.5) * 2

        # Random translation direction.
        rand_vector = np.random.rand(3)
        rand_vector = rand_vector / np.linalg.norm(rand_vector)

        # Perform translation.
        translation_vector = rand_vector * self._step_size * rand
        targ_comp = self._translate_atoms_along_vector(
            mol=targ_comp,
            vector=translation_vector,
        )

        # Define a random rotation of the guest.
        # Random number from -1 to 1 for multiplying rotation.
        rand = (random.random() - 0.5) * 2
        rotation_angle = self._rotation_step_size * rand
        rand_axis = np.random.rand(3)
        rand_axis = rand_axis / np.linalg.norm(rand_vector)

        # Perform rotation.
        targ_comp = self._rotate_atoms_by_angle(
            mol=targ_comp,
            angle=rotation_angle,
            axis=rand_axis,
            origin=targ_comp.get_centroid(),
        )

        component_list[targ_comp_id] = targ_comp
        supramolecule = SupraMolecule.init_from_components(
            components=component_list,
        )

        nonbonded_potential = self._compute_potential(supramolecule)
        return supramolecule, nonbonded_potential

    def _stable_dynamics(self, blob: Blob) -> Blob:
        """

        """

        # This is effectively a spindry run.

        raise NotImplementedError()

    def grow_blob(self, host: Host) -> abc.Iterable[StepResult]:
        """
        Grow blob from beads inside host.

        Parameters:

            host:
                The supramolecule to optimize.

        Yields:

            step_result:
                The result of this step.

        """

        # Define single bead blob at host centroid.
        blob = Blob(
            beads=(self._bead_type, ),
            position_matrix=np.array((host.get_centroid(), )),
        )
        step_result = StepResult(
            0,
            potential=self._compute_potential(host, blob),
            blob=blob,
        )
        for step in range(1, self._num_steps):
            # Modify Blob.
            # Add bead to blob.
            blob = blob.with_new_bead(self._bead_type)
            # Reduce Blob.
            blob = blob.reduce_blob()

            # Test new configuration.
            # Perform stable rigid-body dynamics, optimise config.
            blob = self._stable_dynamics(blob)
            # Calculate new potential.
            new_potential = self._compute_potential(host, blob)

            # Perform pull rigid-body dynamics in N directions.

            step_result = StepResult(
                step,
                potential=new_potential,
                blob=blob,
            )
            print(step_result)
            import sys
            sys.exit()
            yield step_result
        import sys
        sys.exit()

        cid = 0
        nonbonded_potential = self._compute_potential(supramolecule)

        yield SupraMolecule(
            atoms=supramolecule.get_atoms(),
            bonds=supramolecule.get_bonds(),
            position_matrix=supramolecule.get_position_matrix(),
            cid=cid,
            potential=nonbonded_potential,
        )
        cids_passed = [cid]
        for step in range(1, self._max_attempts):
            n_supramolecule, n_nonbonded_potential = self._run_step(
                supramolecule=supramolecule,
            )
            passed = self._test_move(
                curr_pot=nonbonded_potential,
                new_pot=n_nonbonded_potential
            )
            if passed:
                cid += 1
                cids_passed.append(cid)
                nonbonded_potential = n_nonbonded_potential
                supramolecule = SupraMolecule(
                    atoms=supramolecule.get_atoms(),
                    bonds=supramolecule.get_bonds(),
                    position_matrix=(
                        n_supramolecule.get_position_matrix()
                    ),
                    cid=cid,
                    potential=nonbonded_potential,
                )

                yield supramolecule

            if len(cids_passed) == self._num_conformers:
                break

        if verbose:
            print(
                f'{len(cids_passed)} conformers generated in {step} '
                'steps.'
            )

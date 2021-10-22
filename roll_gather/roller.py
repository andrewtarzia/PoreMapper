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
        bead_sigma: float,
        max_size_modifier: float,
        max_beads: int,
        num_steps: int,
        num_dynamics_steps: int,
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

            bead_sigma:
                Bead sigma to use in Blob.

            max_size_modifier:
                Maximum percent to modify blob by.

            max_beads:
                Maximum number of beads in Blob.

            num_steps:
                Number of steps to run growth for.

            num_dynamics_steps:
                Number of steps for each dynamics run.

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
        self._bead_sigma = bead_sigma
        self._max_size_modifier = max_size_modifier
        self._max_beads = max_beads
        self._num_steps = num_steps
        self._num_dynamics_steps = num_dynamics_steps
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
    ) -> float:
        """
        Define a Lennard-Jones nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (self._bead_sigma/distance) ** 12
                - (self._bead_sigma/distance) ** 6
            )
        )

    def _get_distances(self, host: Host, blob: Blob) -> np.ndarray:
        return cdist(
            host.get_position_matrix(),
            blob.get_position_matrix(),
        )

    def _compute_potential(self, host: Host, blob: Blob) -> float:
        nonbonded_potential = 0
        pair_dists = self._get_distances(host, blob)
        nonbonded_potential += np.sum(
            self._nonbond_potential(
                distance=pair_dists,
            )
        )

        return nonbonded_potential

    def _translate_beads_along_vector(
        self,
        blob: Blob,
        vector: np.ndarray,
    ) -> Blob:
        return blob.with_displacement(vector)

    def _modify_blob_extent(self, blob: Blob, percent: float) -> Blob:
        pos_mat = blob.get_position_matrix()
        centroid = blob.get_centroid()
        new_position_matrix = []
        for pos in pos_mat:
            dist_to_centroid = pos-centroid
            modified_pos = centroid+(
                dist_to_centroid*((100-percent)/100)
            )
            new_position_matrix.append(modified_pos)

        return blob.with_position_matrix(
            np.array(new_position_matrix)
        )

    def _rotate_beads_by_angle(
        self,
        blob: Blob,
        angle: float,
        axis: np.ndarray,
        origin: np.ndarray,
    ) -> Blob:
        new_position_matrix = blob.get_position_matrix()
        # Set the origin of the rotation to "origin".
        new_position_matrix = new_position_matrix - origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(angle, axis)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        new_position_matrix = (rot_mat @ new_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        new_position_matrix = new_position_matrix + origin

        return blob.with_position_matrix(new_position_matrix)

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

    def _run_step(self, host, blob):

        # # Perform translation.
        # # Random number from -1 to 1 for multiplying translation.
        # rand = (random.random() - 0.5) * 2
        # # Random translation direction.
        # rand_vector = np.random.rand(3)
        # rand_vector = rand_vector / np.linalg.norm(rand_vector)
        # # Perform translation.
        # translation_vector = rand_vector * self._step_size * rand
        # blob = self._translate_beads_along_vector(
        #     blob=blob,
        #     vector=translation_vector,
        # )

        # Random number from -1 to 1 for multiplying percent.
        rand = (random.random() - 0.5) * 2
        # Perform extent modification.
        percent = self._max_size_modifier * rand
        blob = self._modify_blob_extent(
            blob=blob,
            percent=percent,
        )

        # Define a random rotation of the guest.
        # Random number from -1 to 1 for multiplying rotation.
        rand = (random.random() - 0.5) * 2
        rotation_angle = self._rotation_step_size * rand
        rand_axis = np.random.rand(3)
        rand_vector = np.random.rand(3)
        rand_axis = rand_axis / np.linalg.norm(rand_vector)

        # Perform rotation.
        blob = self._rotate_beads_by_angle(
            blob=blob,
            angle=rotation_angle,
            axis=rand_axis,
            origin=blob.get_centroid(),
        )

        potential = self._compute_potential(host, blob)
        return blob, potential

    def _stable_dynamics(
        self,
        host: Host,
        blob: Blob,
        potential: float
    ) -> tuple[Blob, float]:
        """
        Allow the blob to modify in position, size and orientation.

        """

        # This is effectively a survey of rotations and
        # contractions/expansions only.
        for step in range(self._num_dynamics_steps):
            new_blob, new_potential = self._run_step(host, blob)
            passed = self._test_move(
                curr_pot=potential,
                new_pot=new_potential
            )
            if passed:
                blob = new_blob
                potential = new_potential

        return (blob, potential)

    def grow_blob(self, host: Host) -> abc.Iterable[StepResult]:
        """
        Grow blob from beads inside host.

        Parameters:

            host:
                The host to analyse.

        Yields:

            step_result:
                The result of this step.

        """

        # Define single bead blob at host centroid.
        blob = Blob(
            beads=(Bead(id=0, sigma=self._bead_sigma), ),
            position_matrix=np.array((host.get_centroid(), )),
        )
        potential = self._compute_potential(host, blob)
        step_result = StepResult(
            0,
            potential=self._compute_potential(host, blob),
            blob=blob,
        )
        print(step_result)
        for step in range(1, self._num_steps):
            # Modify Blob.
            # Add bead to blob.
            min_host_guest_distance = min(self._get_distances(
                host=host,
                blob=blob,
            ).flatten())
            blob = blob.with_new_bead(
                min_host_guest_distance=min_host_guest_distance,
            )
            blob.write_xyz_file('temp.xyz')
            # Reduce Blob.
            blob = blob.reduce_blob()

            # Test new configuration.
            # Perform stable rigid-body dynamics, optimise config.
            blob, potential = self._stable_dynamics(
                host=host,
                blob=blob,
                potential=potential,
            )

            # Perform pull rigid-body dynamics in N directions.

            step_result = StepResult(
                step,
                potential=potential,
                blob=blob,
            )
            print(step_result)
            yield step_result
        import sys
        sys.exit()

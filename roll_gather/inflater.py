"""
Inflater
========

#. :class:`.Inflater`

Generator of blob guests using nonbonded interactions and growth.

"""

from __future__ import annotations
from collections import abc
import typing

import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
import random

from .host import Host
from .blob import Blob
from .pore import Pore
from .step_result import InflationStepResult


class Inflater:
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
        num_dynamics_steps: int,
        bond_epsilon: typing.Optional[float] = 1.,
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

            bond_epsilon:
                Value of epsilon used in the bond potential between
                beads.

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
        self._num_dynamics_steps = num_dynamics_steps
        self._bond_epsilon = bond_epsilon
        self._nonbond_epsilon = nonbond_epsilon
        self._beta = beta
        if random_seed is None:
            np.random.seed()
            random.seed()
        else:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def _get_distances(self, host: Host, blob: Blob) -> np.ndarray:
        return cdist(
            host.get_position_matrix(),
            blob.get_position_matrix(),
        )

    def _translate_beads_along_vector(
        self,
        blob: Blob,
        vector: np.ndarray,
        bead_id: typing.Optional[int] = None,
    ) -> Blob:
        if bead_id is None:
            return blob.with_displacement(vector)
        else:
            new_position_matrix = deepcopy(blob.get_position_matrix())
            for bead in blob.get_beads():
                if bead.get_id() != bead_id:
                    continue
                pos = blob.get_position_matrix()[bead.get_id()]
                new_position_matrix[bead.get_id()] = pos - vector

            return blob.with_position_matrix(new_position_matrix)

    def inflate_blob(
        self,
        host: Host,
    ) -> abc.Iterable[InflationStepResult]:
        """
        Mould blob from beads inside host.

        Parameters:

            host:
                The host to analyse.

        Yields:

            The result of this step.

        """

        # for num_beads in range(1, self._max_beads):
        # Define an idealised blob based on num_beads.
        blob = Blob.init_from_idealised_geometry(
            num_beads=self._max_beads,
            bead_sigma=self._bead_sigma,
        )
        blob = blob.with_centroid(host.get_centroid())

        movable_bead_ids = set([i.get_id() for i in blob.get_beads()])
        for step in range(self._num_dynamics_steps):
            for bead in blob.get_beads():
                if bead.get_id() not in movable_bead_ids:
                    continue
                centroid = blob.get_centroid()
                pos_mat = blob.get_position_matrix()
                # Perform translation.
                com_to_bead = pos_mat[bead.get_id()] - centroid
                com_to_bead /= np.linalg.norm(com_to_bead)
                translation_vector = self._step_size * -com_to_bead
                new_blob = self._translate_beads_along_vector(
                    blob=blob,
                    vector=translation_vector,
                    bead_id=bead.get_id(),
                )

                # Check for steric hit.
                min_host_guest_distance = min(self._get_distances(
                    host=host,
                    blob=new_blob,
                ).flatten())
                # If, do not update blob.
                if min_host_guest_distance < self._bead_sigma:
                    movable_bead_ids.remove(bead.get_id())
                else:
                    blob = blob.with_position_matrix(
                        position_matrix=new_blob.get_position_matrix(),
                    )

            num_movable_beads = len(movable_bead_ids)
            if num_movable_beads == blob.get_num_beads():
                nonmovable_bead_ids = [
                    i.get_id() for i in blob.get_beads()
                ]
            else:
                nonmovable_bead_ids = [
                    i.get_id() for i in blob.get_beads()
                    if i.get_id() not in movable_bead_ids
                ]
            pore = Pore(
                blob=blob,
                nonmovable_bead_ids=nonmovable_bead_ids,
            )
            step_result = InflationStepResult(
                step=step,
                num_movable_beads=num_movable_beads,
                blob=blob,
                pore=pore,
            )
            yield step_result

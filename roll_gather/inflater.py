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
        bead_sigma: float,
        num_beads: int,
        num_steps: int,
    ):
        """
        Initialize a :class:`Spinner` instance.

        Parameters:

            step_size:
                The relative size of the step to take during step.

            bead_sigma:
                Bead sigma to use in Blob.

            num_beads:
                Number of beads in Blob.

            num_steps:
                Number of steps to run growth for.

        """

        self._step_size = step_size
        self._bead_sigma = bead_sigma
        self._num_beads = num_beads
        self._num_steps = num_steps

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

        # Define an idealised blob based on num_beads.
        blob = Blob.init_from_idealised_geometry(
            num_beads=self._num_beads,
            bead_sigma=self._bead_sigma,
        )
        blob = blob.with_centroid(host.get_centroid())

        host_maximum_diameter = host.get_maximum_diameter()
        blob_maximum_diameter = blob.get_maximum_diameter()
        movable_bead_ids = set([i.get_id() for i in blob.get_beads()])
        for step in range(self._num_steps):
            # If the distance is further than the maximum diameter.
            # Stop.
            blob_maximum_diameter = blob.get_maximum_diameter()
            if blob_maximum_diameter > host_maximum_diameter:
                yield step_result
                print(f'breaking at step: {step}')
                break
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

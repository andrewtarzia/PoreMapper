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
from .bead import Bead
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

    def _check_steric(
        self,
        host: Host,
        blob: Blob,
        bead: Bead,
    ) -> np.ndarray:

        coord = np.array([blob.get_position_matrix()[bead.get_id()]])
        host_coords = host.get_position_matrix()
        host_radii = np.array([
            i.get_radii() for i in host.get_atoms()
        ]).reshape(host.get_num_atoms(), 1)
        host_bead_distances = cdist(host_coords, coord)
        host_bead_distances += -host_radii
        min_host_guest_distance = np.min(host_bead_distances.flatten())
        if min_host_guest_distance < bead.get_sigma():
            return True
        return False

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
                print(
                    f'breaking at step: {step} with blob larger than '
                    'host'
                )
                break
            if len(movable_bead_ids) == 0:
                yield step_result
                print(
                    f'breaking at step: {step} with no more moveable '
                    'beads'
                )
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
                if_steric_clash = self._check_steric(
                    host=host,
                    blob=new_blob,
                    bead=bead,
                )
                # If, do not update blob.
                if if_steric_clash:
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

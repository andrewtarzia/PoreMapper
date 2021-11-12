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
from .result import InflationResult, InflationStepResult


class Inflater:
    """
    Grow guest blob.

    """

    def __init__(self, bead_sigma: float):
        """
        Initialize a :class:`Inflater` instance.

        Parameters:

            bead_sigma:
                Bead sigma to use in Blob.

        """

        self._bead_sigma = bead_sigma

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

        starting_radius = 0.1
        num_steps = 100

        # Move host to origin.
        host = host.with_centroid([0., 0., 0.])
        host_pos_mat = host.get_position_matrix()
        host_maximum_diameter = host.get_maximum_diameter()
        host_radii_arr = np.array([
            i.get_radii() for i in host.get_atoms()
        ]).reshape(1, host.get_num_atoms())

        # Get num beads and step size based on maximum diameter of
        # host. Using pyWindow code.
        host_radius = host_maximum_diameter / 2
        host_surface_area = 4 * np.pi * host_radius**2
        num_beads = int(np.log10(host_surface_area) * 250)
        step_size = (host_radius-starting_radius)/num_steps

        # Define an idealised blob based on num_beads.
        blob = Blob.init_from_idealised_geometry(
            bead_sigma=self._bead_sigma,
            num_beads=num_beads,
            sphere_radius=starting_radius,
        )
        blob = blob.with_centroid(host.get_centroid())

        blob_maximum_diameter = blob.get_maximum_diameter()
        movable_bead_ids = set([i.get_id() for i in blob.get_beads()])
        for step in range(num_steps):
            # If the distance is further than the maximum diameter.
            # Stop.
            blob_maximum_diameter = blob.get_maximum_diameter()
            if blob_maximum_diameter > host_maximum_diameter:
                print(
                    f'Pop! breaking at step: {step} with blob larger '
                    'than host'
                )
                break
            if len(movable_bead_ids) == 0:
                print(
                    f'breaking at step: {step} with no more moveable '
                    'beads'
                )
                break

            pos_mat = blob.get_position_matrix()

            # Check for steric clashes.
            # Get host-blob distances.
            pair_dists = cdist(pos_mat, host_pos_mat)
            # Include host atom radii.
            pair_dists += -host_radii_arr
            min_pair_dists = np.min(pair_dists, axis=1)
            # Update movable array.
            movable_bead_arr = np.where(
                min_pair_dists < self._bead_sigma, 0, 1
            ).reshape(num_beads, 1)

            # And ids.
            movable_bead_ids = set(
                np.argwhere(movable_bead_arr==1)[:, 0]
            )
            # Update blob.
            blob = blob.with_movable_bead_ids(
                movable_bead_ids=movable_bead_ids,
            )

            # Define step array based on collisions.
            step_arr = movable_bead_arr * step_size
            # Get translations.
            translation_mat = step_arr * (
                pos_mat / np.linalg.norm(
                    x=pos_mat,
                    axis=1,
                ).reshape(num_beads, 1)
            )
            new_pos_mat = pos_mat + translation_mat

            # Do move.
            blob = blob.with_position_matrix(new_pos_mat)

            num_movable_beads = len(movable_bead_ids)
            if num_movable_beads < 0.6*blob.get_num_beads():
                nonmovable_bead_ids = [
                    i.get_id() for i in blob.get_beads()
                    if i.get_id() not in movable_bead_ids
                ]
            else:
                nonmovable_bead_ids = [
                    i.get_id() for i in blob.get_beads()
                    # if i.get_id() not in movable_bead_ids
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

        step_result = InflationStepResult(
            step=step,
            num_movable_beads=num_movable_beads,
            blob=blob,
            pore=pore,
        )
        yield step_result

    def get_inflated_blob(
        self,
        host: Host,
    ) -> InflationResult:
        """
        Mould blob from beads inside host.

        Parameters:

            host:
                The host to analyse.

        Returns:

            The final result of inflation.

        """

        for step_result in self.inflate_blob(host):
            continue

        return InflationResult(
            step=step_result.step,
            num_movable_beads=step_result.num_movable_beads,
            blob=step_result.blob,
            pore=step_result.pore,
        )

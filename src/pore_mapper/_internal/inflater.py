from __future__ import annotations

from collections import abc

import numpy as np
from scipy.spatial.distance import cdist

from .blob import Blob
from .host import Host
from .pore import Pore
from .result import InflationResult, InflationStepResult


class Inflater:
    """
    Grow guest blob.

    """

    def __init__(
        self,
        bead_sigma: float,
        centroid: np.ndarray,
    ):
        """
        Initialize a :class:`Inflater` instance.

        Parameters:

            bead_sigma:
                Bead sigma to use in Blob.

            centroid:
                Position to Inflate Blob at.

        """

        self._bead_sigma = bead_sigma
        self._centroid = centroid

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

        host_pos_mat = host.get_position_matrix()
        host_maximum_diameter = host.get_maximum_diameter()
        host_radii_arr = np.array(
            [i.get_radii() for i in host.get_atoms()]
        ).reshape(1, host.get_num_atoms())

        # Get num beads and step size based on maximum diameter of
        # host. Using pyWindow code.
        host_radius = host_maximum_diameter / 2
        host_surface_area = 4 * np.pi * host_radius**2
        num_beads = int(np.log10(host_surface_area) * 250)
        step_size = (host_radius - starting_radius) / num_steps

        # Define an idealised blob based on num_beads.
        blob = Blob.init_from_idealised_geometry(
            bead_sigma=self._bead_sigma,
            num_beads=num_beads,
            sphere_radius=starting_radius,
        )
        blob = blob.with_centroid(self._centroid)

        blob_maximum_diameter = blob.get_maximum_diameter()
        movable_bead_ids = set([i.get_id() for i in blob.get_beads()])
        for step in range(num_steps):
            # If the distance is further than the maximum diameter.
            # Stop.
            blob_maximum_diameter = blob.get_maximum_diameter()
            if blob_maximum_diameter > host_maximum_diameter:
                break
            if len(movable_bead_ids) == 0:
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
            movable_bead_ids = set(np.argwhere(movable_bead_arr == 1)[:, 0])
            # Update blob.
            blob = blob.with_movable_bead_ids(movable_bead_ids)

            # Define step array based on collisions.
            step_arr = movable_bead_arr * step_size

            # Get translations.
            # This is how far to move the position matrix, based on how
            # far each point is from the centroid of the blob.
            translation_mat = step_arr * (
                (pos_mat - self._centroid)
                / np.linalg.norm(
                    x=pos_mat - self._centroid,
                    axis=1,
                ).reshape(num_beads, 1)
            )

            # Do move.
            new_pos_mat = pos_mat + translation_mat
            blob = blob.with_position_matrix(new_pos_mat)

            num_movable_beads = len(movable_bead_ids)
            if num_movable_beads < 0.6 * blob.get_num_beads():
                nonmovable_bead_ids = [
                    i.get_id()
                    for i in blob.get_beads()
                    if i.get_id() not in movable_bead_ids
                ]
            else:
                nonmovable_bead_ids = [
                    i.get_id()
                    for i in blob.get_beads()
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

"""
Blob
====

#. :class:`.Blob`

Blob class for optimisation.

"""

from __future__ import annotations
from collections import abc

import random
import numpy as np

from .bead import Bead
from .utilities import sample_spherical


class Blob:
    """
    Representation of a Blob containing beads and positions.

    """

    def __init__(
        self,
        beads: abc.Iterable[Bead],
        position_matrix: np.ndarray,
    ):
        """
        Initialize a :class:`Blob` instance from beads.

        Parameters:

            beads:
                Beads that define the blob.

            position_matrix:
                A ``(n, 3)`` matrix holding the position of every atom in
                the :class:`.Molecule`.

        """

        self._beads = tuple(beads)
        self._sigma = self._beads[0].get_sigma()
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )

    def get_sigma(self) -> float:
        """
        Return sigma of beads.

        """

        return self._sigma

    def get_position_matrix(self) -> np.ndarray:
        """
        Return a matrix holding the bead positions.

        Returns:

            The array has the shape ``(n, 3)``. Each row holds the
            x, y and z coordinates of an atom.

        """

        return np.array(self._position_matrix.T)

    def get_centroid(self) -> np.ndarray:
        """
        Return the centroid.

        Returns:

            The centroid of atoms specified by `atom_ids`.

        """

        n_beads = len(self._beads)
        return np.divide(
            self._position_matrix[:, range(n_beads)].sum(axis=1),
            n_beads
        )

    def get_num_beads(self) -> int:
        """
        Return the number of beads.

        """

        return len(self._beads)

    def get_beads(self) -> abc.Iterable[Bead]:
        """
        Yield the beads in the molecule, ordered as input.

        Yields:

            A Bead in the blob.

        """

        for bead in self._beads:
            yield bead

    def with_displacement(self, displacement: np.ndarray) -> Blob:
        """
        Return a displaced clone Blob.

        Parameters:

            displacement:
                The displacement vector to be applied.

        """

        new_position_matrix = (
            self._position_matrix.T + displacement
        )
        return Blob(
            beads=self._beads,
            position_matrix=np.array(new_position_matrix),
        )

    def with_position_matrix(
        self,
        position_matrix: np.ndarray,
    ) -> Blob:
        """
        Return clone Blob with new position matrix.

        Parameters:

            position_matrix:
               A position matrix of the clone. The shape of the
               matrix is ``(n, 3)``.
        """

        clone = self.__class__.__new__(self.__class__)
        Blob.__init__(
            self=clone,
            beads=self._beads,
            position_matrix=np.array(position_matrix),
        )
        return clone

    def with_new_bead(
        self,
        min_host_guest_distance: float,
    ) -> Blob:
        """
        Return clone Blob with new bead.

        Parameters:

            min_host_guest_distance:
                Minimum distance between blob and host.

        Returns:

            Blob with new bead.

        """

        pos_mat = self.get_position_matrix()

        # Pick a direction randomly from a sphere.
        vec = sample_spherical(1)
        # Multiply by host-guest distance /2.
        vec = vec * (min_host_guest_distance / 2)
        placement_vec = self.get_centroid() + vec.T

        # Place bead.
        new_beads = self._beads + (Bead(
            id=self._beads[-1].get_id()+1,
            sigma=self._sigma,
        ), )
        new_position_matrix = np.vstack([
            pos_mat, placement_vec
        ])

        new_blob = self.__class__.__new__(self.__class__)
        Blob.__init__(
            self=new_blob,
            beads=new_beads,
            position_matrix=np.array(new_position_matrix),
        )

        return new_blob

    def reduce_blob(self) -> Blob:
        """
        Return clone Blob with only convex hull beads.

        Returns:

            Reduced blob.

        """

        print('reduction not implemented yet.')
        return self

    def _write_xyz_content(self) -> str:
        """
        Write basic `.xyz` file content of Blob.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, bead in enumerate(self.get_beads(), 1):
            x, y, z = (i for i in coords[bead.get_id()])
            content.append(
                f'B {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\nBlob!\n'

        return content

    def write_xyz_file(self, path) -> None:
        """
        Write blob to path.

        """

        content = self._write_xyz_content()

        with open(path, 'w') as f:
            f.write(''.join(content))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(list(self._beads))} beads)'
        )

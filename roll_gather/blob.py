"""
Blob
====

#. :class:`.Blob`

Blob class for optimisation.

"""

from __future__ import annotations
from collections import abc


# import networkx as nx
# from .molecule import Molecule
import numpy as np

from .bead import Bead


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

    def get_beads(self) -> abc.Iterable[Bead]:
        """
        Yield the beads in the molecule, ordered as input.

        Yields:

            A Bead in the blob.

        """

        for bead in self._beads:
            yield bead

    def with_position_matrix(
        self,
        position_matrix: np.ndarray,
    ) -> Blob:
        """
        Return clone Blob with new position matrix.

        Parameters:

            position_matrix:
               A position matrix of the clone. The shape of the matrix
               is ``(n, 3)``.
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
        bead_type: Bead,
    ) -> Blob:
        """
        Return clone Blob with new bead.

        Parameters:

            beads:
                Beads that define the blob.

        Returns:

            Blob with new bead.

        """

        # Define new position?? -- within XX distance from a bead.

        raise NotImplementedError()
        return new_blob

    def reduce_blob(self) -> Blob:
        """
        Return clone Blob with only convex hull beads.

        Returns:

            Reduced blob.

        """

        raise NotImplementedError()
        return new_blob

    def _write_xyz_content(self) -> str:
        """
        Write basic `.xyz` file content of Blob.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(
                f'B {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\ncid:{self._cid}, pot: {self._potential}\n'

        return content

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{len(list(self._beads))} beads)'
        )

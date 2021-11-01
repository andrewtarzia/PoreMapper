"""
Pore
====

#. :class:`.Pore`

Pore class for optimisation.

"""

from __future__ import annotations
from collections import abc

from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.distance import euclidean
import json

from .bead import Bead
from .blob import Blob


@dataclass
class PoreProperties:
    """
    Data of a Pore.

    """

    num_beads: int
    potential: float
    radius_of_gyration: float
    sphericity: float
    maximum_diameter: float


class Pore:
    """
    Representation of a Pore containing beads and positions.

    """

    def __init__(
        self,
        blob: Blob,
        nonmovable_bead_ids: abc.Iterable[int],
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

        beads = []
        positions = []
        pos_mat = blob.get_position_matrix()
        count = 0
        for bead in blob.get_beads():
            if bead.get_id() in nonmovable_bead_ids:
                beads.append(Bead(id=count, sigma=bead.get_sigma()))
                positions.append(pos_mat[bead.get_id()])
                count += 1

        self._beads = tuple(beads)
        self._sigma = self._beads[0].get_sigma()
        self._num_beads = len(self._beads)
        self._position_matrix = np.array(
            np.array(positions).T,
            dtype=np.float64,
        )

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

        return self._num_beads

    def get_beads(self) -> abc.Iterable[Bead]:
        """
        Yield the beads in the molecule, ordered as input.

        Yields:

            A Bead in the blob.

        """

        for bead in self._beads:
            yield bead

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

    def get_maximum_diameter(self) -> float:
        """
        Return the maximum diameter.

        This method does not account for the van der Waals radius of
        atoms.

        """

        coords = self._position_matrix
        return float(euclidean(coords.min(axis=1), coords.max(axis=1)))

    def get_properties(self, potential: float) -> PoreProperties:

        radius_of_gyration = 0
        sphericity = 0

        return PoreProperties(
            num_beads=self._num_beads,
            potential=potential,
            radius_of_gyration=radius_of_gyration,
            sphericity=sphericity,
            maximum_diameter=self.get_maximum_diameter(),
        )

    def write_properties(self, path: str, potential: float) -> None:
        """
        Write properties as json to path.

        """

        with open(path, 'w') as f:
            json.dump(asdict(self.get_properties(potential)), f)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self._num_beads} beads)'
        )
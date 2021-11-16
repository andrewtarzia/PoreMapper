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
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
import json

from .bead import Bead
from .blob import Blob
from .utilities import get_tensor_eigenvalues


@dataclass
class PoreProperties:
    num_beads: int
    max_dist_to_com: float
    mean_dist_to_com: float
    volume: float
    windows: abc.Iterable[float]


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

            blob:
                Blob defining the pore.

            nonmovable_bead_ids:
                Bead ids in blob that are not movable, and part of
                Pore.

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

        self._blob = blob
        self._beads = tuple(beads)
        self._sigma = self._beads[0].get_sigma()
        self._num_beads = len(self._beads)
        self._position_matrix = np.array(
            np.array(positions).T,
            dtype=np.float64,
        )

    @classmethod
    def init(
        cls,
        blob: Blob,
        num_beads: int,
        sigma: float,
        beads: abc.Iterable[Bead],
        position_matrix: np.ndarray,
    ) -> Pore:
        """
        Initalise an empty pore.

        Parameters:

            blob:
                Blob that defines the pore.

            num_beads:
                Number of beads in pore.

            sigma:
                Sigma of beads in pore.

            beads:
                Beads in pore.

            position_matrix:
                A ``(n, 3)`` matrix holding the position of every atom in
                the :class:`.Molecule`.

        """

        pore = cls.__new__(cls)
        pore._blob = blob
        pore._num_beads = num_beads
        pore._sigma = sigma
        pore._beads = beads
        pore._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )
        return pore

    def get_blob(self) -> Blob:
        return self._blob

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
        content[0] = f'{i}\nPore!\n'

        return content

    def write_xyz_file(self, path) -> None:
        """
        Write blob to path.

        """

        content = self._write_xyz_content()

        with open(path, 'w') as f:
            f.write(''.join(content))


    def get_mean_distance_to_com(self) -> float:
        """
        Return the mean distance to COM.

        This method does not account for the van der Waals radius of
        atoms.

        """

        return np.mean([
            euclidean(i, self.get_centroid())
            for i in self._position_matrix.T
        ])

    def get_maximum_distance_to_com(self) -> float:
        """
        Return the maximum distance to COM.

        This method does not account for the van der Waals radius of
        atoms.

        """

        return max((
            euclidean(i, self.get_centroid())
            for i in self._position_matrix.T
        ))

    def get_volume(self) -> float:
        """
        Gets the volume of the convex hull of the pore.

        This method does not account for the van der Waals radius of
        atoms.

        """

        if self.get_num_beads() < 4:
            return 0.
        else:
            # Scale the positions to include radii of bead.
            coordinates = self.get_position_matrix()
            lengths_ = np.linalg.norm(coordinates, axis=1)
            lengths_and = lengths_ + self._sigma
            scales_ = lengths_and/lengths_
            coordinates = self.get_position_matrix()*scales_.reshape(
                len(coordinates), 1
            )
            return ConvexHull(coordinates).volume

    def get_properties(self) -> PoreProperties:

        return PoreProperties(
            num_beads=self._num_beads,
            max_dist_to_com=self.get_max_dist_to_com(),
            mean_dist_to_com=self.get_mean_dist_to_com(),
            volume=self.get_volume(),
            windows=self.get_windows(),
        )

    def get_windows(self) -> abc.Iterable[float]:
        return self._blob.get_windows()

    def get_inertia_tensor(self) -> np.ndarray:
        """
        Return the tensor of inertia a molecule.

        Returns:

            The tensor of inertia of a molecule.

        """

        coordinates = self.get_position_matrix()
        pow2 = coordinates**2

        diag_1 = np.sum((pow2[:, 1] + pow2[:, 2]))
        diag_2 = np.sum((pow2[:, 0] + pow2[:, 2]))
        diag_3 = np.sum((pow2[:, 0] + pow2[:, 1]))

        mxy = np.sum(-1 * coordinates[:, 0] * coordinates[:, 1])
        mxz = np.sum(-1 * coordinates[:, 0] * coordinates[:, 2])
        myz = np.sum(-1 * coordinates[:, 1] * coordinates[:, 2])

        inertia_tensor = np.array(
            [
                [diag_1, mxy, mxz],
                [mxy, diag_2, myz],
                [mxz, myz, diag_3],
            ]
        ) / coordinates.shape[0]
        return (inertia_tensor)

    def get_asphericity(self):
        S = get_tensor_eigenvalues(
            T=self.get_inertia_tensor(),
            sort=True,
        )
        return (S[0] - (S[1] + S[2]) / 2)

    def get_relative_shape_anisotropy(self):
        S = get_tensor_eigenvalues(
            T=self.get_inertia_tensor(),
            sort=True,
        )
        return (1 - 3 * (
            (S[0] * S[1] + S[0] * S[2] + S[1] * S[2]) / (np.sum(S)
        )**2))

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

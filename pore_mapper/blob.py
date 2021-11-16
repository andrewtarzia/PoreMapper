"""
Blob
====

#. :class:`.Blob`

Blob class for optimisation.

"""

from __future__ import annotations

from collections import abc

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import MeanShift
import json

from .bead import Bead


@dataclass
class BlobProperties:
    num_beads: int
    maximum_diameter: float


class Blob:
    """
    Representation of a Blob containing beads and positions.

    """

    def __init__(
        self,
        beads: abc.Iterable[Bead],
        position_matrix: np.ndarray,
        movable_bead_ids: Optional[abc.Iterable[int]] = None,
    ):
        """
        Initialize a :class:`Blob` instance from beads.

        Parameters:

            beads:
                Beads that define the blob.

            position_matrix:
                A ``(n, 3)`` matrix holding the position of every atom in
                the :class:`.Molecule`.

            movable_bead_ids:
                IDs of beads that are movable.

        """

        self._beads = tuple(beads)
        self._sigma = self._beads[0].get_sigma()
        self._num_beads = len(self._beads)
        if movable_bead_ids is None:
            self._movable_bead_ids = tuple(i.get_id() for i in beads)
        else:
            self._movable_bead_ids = tuple(movable_bead_ids)

        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )

    @classmethod
    def init_from_idealised_geometry(
        cls,
        bead_sigma: float,
        num_beads: int,
        sphere_radius: float,
    ) -> Blob:
        """
        Initalise a blob in an idealised geometry.

        """

        blob = cls.__new__(cls)
        blob._num_beads = num_beads
        blob._sigma = bead_sigma
        blob._beads = tuple(
            Bead(i, bead_sigma) for i in range(num_beads)
        )
        blob._movable_bead_ids = tuple(i.get_id() for i in blob._beads)
        blob._define_idealised_geometry(num_beads, sphere_radius)
        return blob

    def _define_idealised_geometry(
        self,
        num_beads: int,
        sphere_radius: float
    ) -> None:
        """
        Define a sphere with num_beads and radius 0.1.

        Here I use code by Alexandre Devert for spreading points on a
        sphere: http://blog.marmakoide.org/?p=1

        Same code as pywindow.

        """

        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(num_beads)
        z = np.linspace(
            1 - 1.0 / num_beads,
            1.0 / num_beads - 1.0,
            num_beads,
        )
        radius = np.sqrt(1 - z * z)
        points = np.zeros((3, num_beads))
        points[0, :] = sphere_radius * np.cos(theta) * radius
        points[1, :] = sphere_radius * np.sin(theta) * radius
        points[2, :] = z * sphere_radius

        self._position_matrix = np.array(
            points,
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

        return self._num_beads

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
            movable_bead_ids=self._movable_bead_ids,
        )

    def with_centroid(self, position: np.ndarray) -> Blob:
        """
        Return a clone with a new centroid.

        """

        centroid = self.get_centroid()
        displacement = position-centroid
        return self.with_displacement(displacement)

    def get_movable_bead_ids(self):
        return self._movable_bead_ids

    def with_movable_bead_ids(
        self,
        movable_bead_ids: abc.Iterable[int],
    ) -> Blob:
        """
        Return a clone with new movable bead ids.

        """

        clone = self.__class__.__new__(self.__class__)
        Blob.__init__(
            self=clone,
            beads=self._beads,
            position_matrix=self._position_matrix.T,
            movable_bead_ids=movable_bead_ids,
        )
        return clone

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
            movable_bead_ids=self._movable_bead_ids,
        )
        return clone

    def _write_xyz_content(self) -> str:
        """
        Write basic `.xyz` file content of Blob.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, bead in enumerate(self.get_beads(), 1):
            x, y, z = (i for i in coords[bead.get_id()])
            movable = (
                1 if bead.get_id() in self._movable_bead_ids
                else 0
            )
            content.append(
                f'B {x:f} {y:f} {z:f} {movable}\n'
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

    def get_properties(self) -> BlobProperties:

        return BlobProperties(
            num_beads=self._num_beads,
            maximum_diameter=self.get_maximum_diameter(),
        )

    def get_windows(self) -> abc.Iterable[float]:

        if len(self._movable_bead_ids) == self._num_beads:
            return [0]

        if len(self._movable_bead_ids) == 0:
            return [0]

        movable_bead_coords = np.array([
            self._position_matrix.T[i] for i in self._movable_bead_ids
        ])

        # Cluster points.
        clustering = MeanShift().fit(movable_bead_coords)
        labels = set(clustering.labels_)
        windows = []
        for label in labels:
            bead_ids = tuple(
                _id for i, _id in enumerate(self._movable_bead_ids)
                if clustering.labels_[i] == label
            )
            label_coords = np.array([
                self._position_matrix.T[i] for i in bead_ids
            ])
            label_centroid = np.divide(
                label_coords.sum(axis=0), len(bead_ids)
            )
            max_label_distance = max([
                euclidean(i, label_centroid)
                for i in label_coords
            ])
            windows.append(max_label_distance)

        return windows

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

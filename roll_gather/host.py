"""
Host
=====

#. :class:`.Host`

Host class for calculation.

"""

from __future__ import annotations
from collections import abc
import typing

from .atom import Atom
import numpy as np


class Host:
    """
    Representation of a molecule containing atoms and positions.

    """

    def __init__(
        self,
        atoms: abc.Iterable[Atom],
        position_matrix: np.ndarray,
    ):
        """
        Initialize a :class:`Host` instance.

        Parameters:

            atoms:
                Atoms that define the molecule.

            position_matrix:
                A ``(n, 3)`` matrix holding the position of every
                atom in the :class:`.Molecule`.

        """

        self._atoms = tuple(atoms)
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )

    @classmethod
    def init_from_xyz_file(cls, path) -> Host:
        """
        Initialize from a file.

        Parameters:

            path:
                The path to a molecular ``.xyz`` file.

        Returns:

            The host.

        """

        with open(path, 'r') as f:
            _, _, *content = f.readlines()

        atoms = []
        positions = []
        for i, line in enumerate(content):
            element, *coords = line.split()
            positions.append([float(i) for i in coords])

            atoms.append(Atom(id=i, element_string=element))

        return cls(
            atoms=atoms,
            position_matrix=np.array(positions),
        )

    def get_position_matrix(self) -> np.ndarray:
        """
        Return a matrix holding the atomic positions.

        Returns:

            The array has the shape ``(n, 3)``. Each row holds the
            x, y and z coordinates of an atom.

        """

        return np.array(self._position_matrix.T)

    def get_atoms(self) -> abc.Iterable[Atom]:
        """
        Yield the atoms in the molecule, ordered as input.

        Yields:

            An atom in the molecule.

        """

        for atom in self._atoms:
            yield atom

    def get_num_atoms(self) -> int:
        """
        Return the number of atoms in the molecule.

        """

        return len(self._atoms)

    def get_centroid(
        self,
        atom_ids: typing.Optional[abc.Iterable[int]] = None,
    ) -> np.ndarray:
        """
        Return the centroid.

        Parameters:

            atom_ids:
                The ids of atoms which are used to calculate the
                centroid. Can be a single :class:`int`, if a single
                atom is to be used, or ``None`` if all atoms are to
                be used.

        Returns:

            The centroid of atoms specified by `atom_ids`.

        Raises:

            If `atom_ids` has a length of ``0``.

        """

        if atom_ids is None:
            atom_ids = range(len(self._atoms))
        elif isinstance(atom_ids, int):
            atom_ids = (atom_ids, )
        elif not isinstance(atom_ids, (list, tuple)):
            atom_ids = list(atom_ids)

        if len(atom_ids) == 0:
            raise ValueError('atom_ids was of length 0.')

        return np.divide(
            self._position_matrix[:, atom_ids].sum(axis=1),
            len(atom_ids)
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}({len(self._atoms)} atoms) '
            f'at {id(self)}>'
        )

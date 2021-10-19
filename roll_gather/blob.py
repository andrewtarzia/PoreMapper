"""
SupraMolecule
=============

#. :class:`.SupraMolecule`

SupraMolecule class for optimisation.

"""

import networkx as nx
from .molecule import Molecule
import numpy as np


class SupraMolecule(Molecule):
    """
    Representation of a supramolecule containing atoms and positions.

    """

    def __init__(
        self,
        atoms,
        bonds,
        position_matrix,
        cid=None,
        potential=None,
    ):
        """
        Initialize a :class:`Supramolecule` instance.

        Parameters
        ----------
        atoms : :class:`iterable` of :class:`.Atom`
            Atoms that define the molecule.

        bonds : :class:`iterable` of :class:`.Bond`
            Bonds between atoms that define the molecule.

        position_matrix : :class:`numpy.ndarray`
            A ``(n, 3)`` matrix holding the position of every atom in
            the :class:`.Molecule`.

        cid : :class:`int`, optional
            Conformer id of supramolecule.

        potential : :class:`float`, optional
            Potential energy of Supramolecule.

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )
        self._define_components()
        self._cid = cid
        self._potential = potential

    @classmethod
    def init_from_components(cls, components):
        """
        Initialize a :class:`Supramolecule` instance from components.

        Parameters
        ----------
        components : :class:`iterable` of :class:`.Molecule`
            Molecular components that define the supramolecule.

        """

        atoms = []
        bonds = []
        position_matrix = []
        for comp in components:
            for a in comp.get_atoms():
                atoms.append(a)
            for b in comp.get_bonds():
                bonds.append(b)
            for pos in comp.get_position_matrix():
                position_matrix.append(pos)

        supramolecule = cls.__new__(cls)
        supramolecule._atoms = tuple(atoms)
        supramolecule._bonds = tuple(bonds)
        supramolecule._components = tuple(components)
        supramolecule._cid = None
        supramolecule._potential = None
        supramolecule._position_matrix = np.array(position_matrix).T
        return supramolecule

    def _define_components(self):
        """
        Define disconnected component molecules as :class:`.Molecule`s.

        """

        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        for atom in self.get_atoms():
            mol_graph.add_node(atom.get_id())

        # Add edges.
        for bond in self._bonds:
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        comps = []
        for c in nx.connected_components(mol_graph):
            in_atoms = [
                i for i in self._atoms
                if i.get_id() in c
            ]
            in_bonds = [
                i for i in self._bonds
                if i.get_atom1_id() in c and i.get_atom2_id() in c
            ]
            new_pos_matrix = self._position_matrix[:, list(c)].T
            comps.append(
                Molecule(in_atoms, in_bonds, new_pos_matrix)
            )

        self._components = tuple(comps)

    def _write_xyz_content(self):
        """
        Write basic `.xyz` file content of Molecule.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\ncid:{self._cid}, pot: {self._potential}\n'

        return content

    def get_components(self):
        """
        Yields each molecular component.

        """

        for i in self._components:
            yield i

    def get_cid(self):
        return self._cid

    def get_potential(self):
        return self._potential

    def __str__(self):
        return repr(self)

    def __repr__(self):
        comps = ', '.join([str(i) for i in self.get_components()])
        return (
            f'{self.__class__.__name__}('
            f'{len(list(self.get_components()))} components, '
            f'{comps})'
        )

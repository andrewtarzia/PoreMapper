"""
Atom
====

#. :class:`.Atom`

Atom class.

"""

from .radii import get_radius


class Atom:
    """
    Atom.

    """

    def __init__(self, id, element_string):
        """
        Initialize a :class:`Atom` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to atom.

        element_string : :class:`str`
            Atom element symbol as string.

        """

        self._id = id
        self._element_string = element_string
        self._radii = get_radius(element_string)

    def get_id(self):
        """
        Get atom ID.

        """

        return self._id

    def get_element_string(self):
        """
        Get atom element symbol.

        """

        return self._element_string

    def get_radii(self):
        """
        Get atomic radii (STREUSEL).

        """

        return self._radii


    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.get_element_string()}('
            f'id={self.get_id()}, radii={self._radii})'
        )
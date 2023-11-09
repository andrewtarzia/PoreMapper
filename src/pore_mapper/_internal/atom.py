from .radii import get_radius


class Atom:
    """
    Atom.

    """

    def __init__(self, id: int, element_string: str) -> None:
        """
        Initialize a :class:`Atom` instance.

        Parameters:

            id:
                ID to be assigned to atom.

            element_string:
                Atom element symbol as string.

        """

        self._id = id
        self._element_string = element_string
        self._radii = get_radius(element_string)

    def get_id(self) -> int:
        """
        Get atom ID.

        """

        return self._id

    def get_element_string(self) -> str:
        """
        Get atom element symbol.

        """

        return self._element_string

    def get_radii(self) -> float:
        """
        Get atomic radii (STREUSEL).

        """

        return self._radii

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return (
            f"{self.get_element_string()}("
            f"id={self.get_id()}, radii={self._radii})"
        )

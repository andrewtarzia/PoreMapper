"""
Bead
====

#. :class:`.Bead`

Bead class.

"""


class Bead:
    """
    Atom.

    """

    def __init__(self, sigma: float):
        """
        Initialize a :class:`Bead` instance.

        Parameters:
            sigma:
                Size (angstrom) of bead.

        """

        self._sigma = sigma

    def get_sigma(self) -> float:
        """
        Get atom sigma.

        """
        return self._sigma

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}(sigma={self._sigma}) '
            f'at {id(self)}>'
        )

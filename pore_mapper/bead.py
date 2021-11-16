"""
Bead
====

#. :class:`.Bead`

Bead class.

"""


class Bead:
    """
    Bead.

    """

    def __init__(self, id: int, sigma: float):
        """
        Initialize a :class:`Bead` instance.

        Parameters:

            id:
                ID of bead.

            sigma:
                Size (angstrom) of bead.

        """

        self._id = id
        self._sigma = sigma

    def get_id(self) -> int:
        """
        Get bead id.

        """
        return self._id

    def get_sigma(self) -> float:
        """
        Get bead sigma.

        """
        return self._sigma

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'<{self.__class__.__name__}(id={self._id}, '
            f'sigma={self._sigma}) '
            f'at {id(self)}>'
        )

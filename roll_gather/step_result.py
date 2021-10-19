"""
Step Result
===========

#. :class:`.StepResult`

Result of one step of calculation.

"""

from dataclasses import dataclass

from .blob import Blob


@dataclass(frozen=True)
class StepResult:
    """
    Data of one step of calculation.

    """
    step: int
    potential: float
    blob: Blob

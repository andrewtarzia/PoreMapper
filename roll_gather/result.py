"""
Result
======

#. :class:`.InflationResult`
#. :class:`.InflationStepResult`

"""

from dataclasses import dataclass

from .blob import Blob
from .pore import Pore


@dataclass(frozen=True)
class InflationResult:
    step: int
    num_movable_beads: float
    blob: Blob
    pore: Pore


@dataclass(frozen=True)
class InflationStepResult:
    step: int
    num_movable_beads: float
    blob: Blob
    pore: Pore

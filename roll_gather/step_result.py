"""
Step Result
===========

#. :class:`.StepResult`

Result of one step of calculation.

"""

from dataclasses import dataclass

from .blob import Blob
from .pore import Pore


@dataclass(frozen=True)
class StepResult:
    step: int
    potential: float
    blob: Blob


@dataclass(frozen=True)
class InflationStepResult:
    step: int
    num_movable_beads: float
    blob: Blob
    pore: Pore

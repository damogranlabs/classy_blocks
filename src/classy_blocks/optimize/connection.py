import dataclasses
from typing import Set


@dataclasses.dataclass
class CellConnection:
    """A connection between two points;
    they are refered by indexes rather than positions"""

    corners: Set[int]  # cell-local indexes
    indexes: Set[int]

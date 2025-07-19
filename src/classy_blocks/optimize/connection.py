import dataclasses


@dataclasses.dataclass
class CellConnection:
    """A connection between two points;
    they are refered by indexes rather than positions"""

    corners: set[int]  # cell-local indexes
    indexes: set[int]

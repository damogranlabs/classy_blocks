"""Misc utilities"""

import dataclasses

from classy_blocks.base.exceptions import CornerPairError
from classy_blocks.types import OrientType
from classy_blocks.util.constants import SIDES_MAP
from classy_blocks.util.frame import Frame


def report(text):
    """TODO: improve (verbosity, logging, ...)"""
    print(text)


def indent(text: str, levels: int) -> str:
    """Indents 'text' by 'levels' tab characters"""
    return "\t" * levels + text + "\n"


@dataclasses.dataclass
class EdgeLocation:
    """A helper class that maps top/bottom/side faces of an operation and corner indexes"""

    corner_1: int
    corner_2: int

    side: OrientType

    @property
    def start_corner(self) -> int:
        """Returns start corner for this location"""
        diff = abs(self.corner_1 - self.corner_2)
        corner_min = min(self.corner_1, self.corner_2)
        corner_max = max(self.corner_1, self.corner_2)

        if diff in (1, 4):
            # neighbours on top/bottom face (diff == 1) or
            # side corners (diff == 4)
            return corner_min % 4

        if diff == 3:
            # the last corner of a face (0...3 or 4...7):
            return corner_max % 4

        raise CornerPairError(f"Given pair: {self.corner_1}-{self.corner_2}")


edge_map = Frame[EdgeLocation]()
for i in range(4):
    corner_1 = i
    corner_2 = (i + 1) % 4

    # bottom face
    edge_map.add_beam(corner_1, corner_2, EdgeLocation(corner_1, corner_2, "bottom"))
    # top face
    edge_map.add_beam(corner_1 + 4, corner_2 + 4, EdgeLocation(corner_1 + 4, corner_2 + 4, "top"))
    # side edges
    edge_map.add_beam(corner_1, corner_1 + 4, EdgeLocation(corner_1, corner_1 + 4, SIDES_MAP[i]))

import dataclasses
from typing import List, Optional, Set

from classy_blocks.optimize.cell import CellBase
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.links import LinkBase
from classy_blocks.types import NPPointListType, NPPointType


class ClampExistsError(Exception):
    """Raised when adding a clamp to a junction that already has one defined"""


@dataclasses.dataclass
class IndexedLink:
    link: LinkBase
    follower_index: int


class Junction:
    """A class that collects Cells/Blocks that
    share the same Vertex"""

    def __init__(self, points: NPPointListType, index: int):
        self.points = points
        self.index = index
        self.cells: Set[CellBase] = set()

        self.clamp: Optional[ClampBase] = None
        self.links: List[IndexedLink] = []

    @property
    def point(self) -> NPPointType:
        return self.points[self.index]

    def add_cell(self, cell: CellBase) -> None:
        """Adds the given cell to the list if it is
        a part of this junction (one common vertex)"""
        for index in cell.indexes:
            if index == self.index:
                self.cells.add(cell)
                return

    def add_clamp(self, clamp: ClampBase) -> None:
        if self.clamp is not None:
            raise ClampExistsError(f"Clamp already defined for junction {self.index}")

        self.clamp = clamp

    def add_link(self, link: LinkBase, follower_index: int) -> None:
        self.links.append(IndexedLink(link, follower_index))

    @property
    def quality(self) -> float:
        """Returns average quality of all cells at this junction;
        this serves as an indicator of which junction to optimize,
        not a measurement of overall mesh quality"""
        return sum([cell.quality for cell in self.cells]) / len(self.cells)

    @property
    def delta(self) -> float:
        """Defining length for calculation of gradients, displacements, etc."""
        return min(cell.min_length for cell in self.cells)

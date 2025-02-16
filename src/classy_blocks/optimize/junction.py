import dataclasses
from typing import List, Optional, Set

from classy_blocks.base.exceptions import ClampExistsError
from classy_blocks.cbtyping import NPPointListType, NPPointType
from classy_blocks.optimize.cell import CellBase
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.links import LinkBase


@dataclasses.dataclass
class IndexedLink:
    # TODO: refactor / deuglify
    link: LinkBase
    follower_index: int


class Junction:
    """A class that collects Cells that
    share the same Vertex"""

    def __init__(self, points: NPPointListType, index: int):
        self.points = points
        self.index = index

        self.cells: Set[CellBase] = set()

        self.neighbours: List[Junction] = []

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

    def add_neighbour(self, to: "Junction") -> bool:
        """Returns True if this Junction is connected to passed one"""
        if to == self:
            return False

        # if any of connections within a cell is equal to
        # the connection between these two junctions,
        # they are connected
        junction_indexes = {self.index, to.index}

        for cell in self.cells:
            for connection in cell.connections:
                if connection.indexes == junction_indexes:
                    if to not in self.neighbours:
                        self.neighbours.append(to)
                        return True

        return False

    def add_clamp(self, clamp: ClampBase) -> None:
        if self.clamp is not None:
            raise ClampExistsError(f"Clamp already defined for junction {self.index}")

        self.clamp = clamp

    def add_link(self, link: LinkBase, follower_index: int) -> None:
        self.links.append(IndexedLink(link, follower_index))

    @property
    def is_boundary(self) -> bool:
        """Returns True if this junction lies on boundary"""
        for cell in self.cells:
            if self.index in cell.boundary:
                return True

        return False

    @property
    def quality(self) -> float:
        """Returns average quality of all cells at this junction;
        this serves as an indicator of which junction to optimize,
        not a measurement of overall mesh quality"""
        return sum([cell.quality for cell in self.cells]) / len(self.cells)

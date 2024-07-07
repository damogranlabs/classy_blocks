from typing import List

from classy_blocks.optimize.cell import Cell
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.junction import Junction
from classy_blocks.types import IndexType, NPPointListType


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class Grid:
    """A list of cells and junctions"""

    def __init__(self, points: NPPointListType, addressing: List[IndexType]):
        # work on a fixed point array and only refer to it instead of building
        # new numpy arrays for every calculation
        self.points = points

        self.cells = [Cell(self.points, indexes) for indexes in addressing]
        self.junctions = [Junction(self.points[index], index) for index in range(len(self.points))]

        self._bind_junctions()
        self._bind_neighbours()

    def _bind_junctions(self) -> None:
        """Adds cells to junctions"""
        for cell in self.cells:
            for junction in self.junctions:
                junction.add_cell(cell)

    def _bind_neighbours(self) -> None:
        """Adds neighbours to cells"""
        for cell_1 in self.cells:
            for cell_2 in self.cells:
                cell_1.add_neighbour(cell_2)

    def get_junction_from_clamp(self, clamp: ClampBase) -> Junction:
        for junction in self.junctions:
            if junction.clamp == clamp:
                return junction

        raise NoJunctionError

    def add_clamp(self, clamp: ClampBase) -> None:
        for junction in self.junctions:
            if junction.index == clamp.vertex.index:
                junction.add_clamp(clamp)

    @property
    def clamps(self) -> List[ClampBase]:
        clamps: List[ClampBase] = []

        for junction in self.junctions:
            if junction.clamp is not None:
                clamps.append(junction.clamp)

        return clamps

    @property
    def quality(self) -> float:
        """Returns summed qualities of all junctions"""
        # It is only called when optimizing linked clamps
        # or at the end of an iteration.
        return sum([cell.quality for cell in self.cells])

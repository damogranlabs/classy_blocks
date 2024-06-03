from typing import Optional, Set

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.cell import Cell
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.types import NPPointListType


class NoClampError(Exception):
    """Raised when this junction has no clamp defined but slope calculation is requested"""


class Junction:
    """A class that collects Cells/Blocks that
    share the same Vertex"""

    def __init__(self, vertex: Vertex, points: NPPointListType):
        self.vertex = vertex
        self.points = points
        self.index = self.vertex.index

        self.cells: Set[Cell] = set()
        self.neighbours: Set[Junction] = set()

        self.clamp: Optional[ClampBase] = None

    def add_cell(self, cell: Cell) -> None:
        """Adds the given cell to the list if it is
        a part of this junction (one common vertex)"""
        for vertex in cell.vertices:
            if vertex == self.vertex:
                self.cells.add(cell)
                return

    def add_neighbour(self, junction: "Junction") -> bool:
        """Adds a junction to the list of neighbours
        if both have at least one common cell;
        returns False otherwise"""
        if junction == self:
            return False

        for cell in junction.cells:
            if cell in self.cells:
                self.neighbours.add(junction)
                return True

        return False

    def add_clamp(self, clamp: ClampBase) -> None:
        self.clamp = clamp

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

    def __hash__(self):
        # to be able to use this object as a dictionary key
        return id(self)

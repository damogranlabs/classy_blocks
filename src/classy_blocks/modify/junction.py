from typing import Optional, Set

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.cell import Cell
from classy_blocks.modify.clamps.clamp import ClampBase


class ClampExistsError(Exception):
    """Raised when adding a clamp to a junction that already has one defined"""


class Junction:
    """A class that collects Cells/Blocks that
    share the same Vertex"""

    def __init__(self, vertex: Vertex):
        self.vertex = vertex
        self.cells: Set[Cell] = set()

        self.clamp: Optional[ClampBase] = None

    def add_cell(self, cell: Cell) -> None:
        """Adds the given cell to the list if it is
        a part of this junction (one common vertex)"""
        for vertex in cell.vertices:
            if vertex == self.vertex:
                self.cells.add(cell)
                return

    def add_clamp(self, clamp: ClampBase) -> None:
        if self.clamp is not None:
            raise ClampExistsError(f"Clamp already defined for junctoin {self.vertex.index}")

        self.clamp = clamp

    @property
    def quality(self) -> float:
        """Returns average quality of all cells at this junction;
        this serves as an indicator of which junction to optimize,
        not a measurement of overall mesh quality"""
        for cell in self.cells:
            cell.invalidate()

        return sum([cell.quality for cell in self.cells]) / len(self.cells)

    @property
    def delta(self) -> float:
        """Defining length for calculation of gradients, displacements, etc."""
        return min(cell.min_length for cell in self.cells)

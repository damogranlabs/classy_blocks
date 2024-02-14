from typing import Set

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.cell import Cell


class Junction:
    """A class that collects Cells/Blocks that
    share the same Vertex"""

    def __init__(self, vertex: Vertex):
        self.vertex = vertex
        self.cells: Set[Cell] = set()

    def add_cell(self, cell: Cell) -> None:
        """Adds the given cell to the list if it is
        a part of this junction (one common vertex)"""
        for vertex in cell.vertices:
            if vertex == self.vertex:
                self.cells.add(cell)
                return

    @property
    def quality(self) -> float:
        """Returns average quality of all cells at this junction;
        this serves as an indicator of which junction to optimize,
        not a measurement of overall mesh quality"""
        return sum([cell.quality for cell in self.cells]) / len(self.cells)

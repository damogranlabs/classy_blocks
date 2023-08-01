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
        """Adds the given cell to the list and returns True if
        it is part of this junction (one common vertex);
        return False otherwise"""
        for vertex in cell.vertices:
            print(f"Checking vertex {vertex.index}")
            if vertex == self.vertex:
                self.cells.add(cell)
                print(f"Adding vertex {self.vertex.index}")
                return

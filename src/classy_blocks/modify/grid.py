from typing import List

from classy_blocks.mesh import Mesh
from classy_blocks.modify.cell import Cell
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.junction import Junction


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class Grid:
    """A list of cells and junctions"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.cells = [Cell(block) for block in self.mesh.blocks]
        self.junctions = [Junction(vertex) for vertex in self.mesh.vertices]

        self._bind_junctions()
        self._bind_cell_neighbours()
        self._bind_junction_neighbours()

    def _bind_junctions(self) -> None:
        """Adds cells to junctions"""
        for cell in self.cells:
            for junction in self.junctions:
                junction.add_cell(cell)

    def _bind_cell_neighbours(self) -> None:
        """Adds neighbours to cells"""
        for cell_1 in self.cells:
            for cell_2 in self.cells:
                cell_1.add_neighbour(cell_2)

    def _bind_junction_neighbours(self) -> None:
        """Adds neighbours to junctions"""
        for junction_1 in self.junctions:
            for junction_2 in self.junctions:
                junction_1.add_neighbour(junction_2)

    def get_junction_from_clamp(self, clamp: ClampBase) -> Junction:
        for junction in self.junctions:
            if junction.clamp == clamp:
                return junction

        raise NoJunctionError

    def add_clamp(self, clamp: ClampBase) -> None:
        for junction in self.junctions:
            if junction.vertex == clamp.vertex:
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
        return sum([cell.quality for cell in self.cells])

    def clear_cache(self):
        for cell in self.cells:
            cell._quality = 0

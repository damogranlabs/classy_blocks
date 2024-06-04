from typing import List

import numpy as np

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

        # store all mesh points in a numpy array for faster
        # calculations; when a vertex position is modified, update the
        # array using update()
        self.points = np.array([vertex.position for vertex in self.mesh.vertices])

        self.cells = [Cell(block, self.points) for block in self.mesh.blocks]
        self.junctions = [Junction(vertex) for vertex in self.mesh.vertices]

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

    def update(self, junction: Junction) -> None:
        self.points[junction.vertex.index] = junction.vertex.position
        for cell in junction.cells:
            cell.invalidate()

        # also update linked stuff
        if junction.clamp is not None:
            if junction.clamp.is_linked:
                linked_junction = self.get_junction_from_clamp(junction.clamp)
                self.points[linked_junction.vertex.index] = linked_junction.vertex.position
                for cell in linked_junction.cells:
                    cell.invalidate()

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
        # It is only called when optimizing linked clamps
        # or at the end of an iteration.
        for cell in self.cells:
            cell.invalidate()

        return sum([cell.quality for cell in self.cells])

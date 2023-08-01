from classy_blocks.mesh import Mesh
from classy_blocks.modify.cell import Cell
from classy_blocks.modify.junction import Junction


class Grid:
    """A list of cells and junctions"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.cells = [Cell(block) for block in self.mesh.blocks]
        self.junctions = [Junction(vertex) for vertex in self.mesh.vertices]

        self._bind_junctions()

    def _bind_junctions(self) -> None:
        """Adds cells to junctions"""
        for cell in self.cells:
            for junction in self.junctions:
                junction.add_cell(cell)

    def _bind_neighbours(self) -> None:
        """Adds neighbours to cells"""

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

    @property
    def quality(self) -> float:
        """Returns summed qualities of all cells in this grid"""
        return sum([cell.quality for cell in self.cells])

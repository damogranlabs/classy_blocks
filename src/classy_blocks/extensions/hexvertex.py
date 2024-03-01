"""Defines a numbered vertex in 3D space and HexCell"""

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import PointType


class HexVertex(Vertex):
    """A Vertex with an added list of of cells to which this vertex attaches"""

    def __init__(self, position: PointType, index: int):
        super().__init__(position, index)

        # add list for cell indices of cells attached to this vertex
        self.cell_indices: list[int] = []

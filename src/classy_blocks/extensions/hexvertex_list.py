from typing import List

from classy_blocks.construct.point import Point
from classy_blocks.extensions.hexvertex import HexVertex
from classy_blocks.types import NPPointType

# this is purely an attempt to make vertexlist (find_unique function) run faster
# this is a clumsy way to do this but it works !!


class HexVertexList:
    """Handling of the 'vertices' part of hexmesh"""

    def __init__(self) -> None:
        self.hexvertices: List[HexVertex] = []

    # def add(self, point: Point) -> HexVertex:
    def add(self, point: Point) -> HexVertex:
        """Add HexVertex - with no duplicate counting"""

        # simply add a new vertex to the list with no checking
        vertex = HexVertex.from_point(point, len(self.hexvertices))
        self.hexvertices.append(vertex)

        return vertex

    @property
    def positions(self) -> List[NPPointType]:
        return [vtx.position for vtx in self.hexvertices]

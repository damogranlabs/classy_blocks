from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.mesh import Mesh
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class FinderBase:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh


class VertexFinder(FinderBase):
    """Ways of finding vertices in a prepared mesh"""

    def by_position(self, position: PointType) -> List[Vertex]:
        """Returns a list of vertices at given position;
        search tolerance is constants.TOL.

        One or two vertices can be at the same spot;
        the latter is the case with internal baffles or face-merged patches.
        Also, no vertices can be found if passed position is off the charts."""
        vertices: List[Vertex] = []

        # TODO: use octree/kdtree if this proves to be too slow
        for vertex in self.mesh.vertices:
            if f.norm(vertex.position - position) <= constants.TOL:
                vertices.append(vertex)

        return vertices


class FaceFinder(FinderBase):
    """Ways of finding faces in a prepared mesh"""


class EdgeFinder(FinderBase):
    """Ways of finding edges in a prepared mesh"""


class BlockFinder(FinderBase):
    """Locating a block in a prepared mesh""" """Locating a block in a prepared mesh"""

from typing import List, Optional

from classy_blocks.items.vertex import Vertex
from classy_blocks.mesh import Mesh
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class VertexFinder:
    """Provides means of finding a Mesh Vertex to be moved"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def by_position(self, position: PointType, radius: Optional[float] = None) -> List[Vertex]:
        """Returns a list of vertices that are
        inside a sphere of given radius; if that is not given,
        constants.TOL is taken"""

        found_vertices: List[Vertex] = []

        if radius is None:
            radius = constants.TOL

        # TODO: optimize with octree/kdtree (not sure if necessary)
        for vertex in self.mesh.vertices:
            if f.norm(vertex.position - position) < radius:
                found_vertices.append(vertex)

        return found_vertices

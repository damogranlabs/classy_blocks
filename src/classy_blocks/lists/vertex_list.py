from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.cbtyping import NPPointType
from classy_blocks.construct.point import Point
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class DuplicatedEntry:
    """A pair vertex:{set of slave patches} that describes
    a duplicated vertex on mentioned patches"""

    def __init__(self, vertex: Vertex, patches: set[str]):
        self.vertex = vertex
        self.patches = patches

    @property
    def point(self) -> NPPointType:
        """Vertex's point"""
        return self.vertex.position


class VertexList:
    """Handling of the 'vertices' part of blockMeshDict"""

    def __init__(self, vertices: list[Vertex]) -> None:
        self.vertices = vertices

        # a collection of duplicated vertices
        # belonging to a certain patch name
        self.duplicated: list[DuplicatedEntry] = []

    def find_duplicated(self, position: NPPointType, slave_patches: set[str]) -> Vertex:
        """Finds an appropriate entry in self.duplicated, if any"""
        # TODO: use vertex indexing
        for dupe in self.duplicated:
            if f.norm(position - dupe.point) < constants.TOL:
                if dupe.patches == slave_patches:
                    return dupe.vertex

        raise VertexNotFoundError(f"No duplicated vertex found: {position} {slave_patches}")

    def add_duplicated(self, point: Point, slave_patches: set[str]) -> Vertex:
        """Re-use existing vertices when there's already one at the position;
        unless that vertex belongs to a slave of a face-merged pair -
        in that case add a duplicate in the same position anyway"""

        # different scenarios:
        # these are covered by grid and KDTree logic
        # 1. add a new vertex, nothing exist at this location yet
        # 2. reuse an existing vertex at this location

        # we need to take care about this:
        # 3. add a new, duplicated vertex at the same location but for a different set of slave patches
        # 4. add a new, 'master' vertex because what's at the same location belongs to a slave patch

        try:
            vertex = self.find_duplicated(point.position, slave_patches)
        except VertexNotFoundError:
            vertex = Vertex.from_point(point, len(self.vertices))
            self.vertices.append(vertex)
            self.duplicated.append(DuplicatedEntry(vertex, slave_patches))

        return vertex

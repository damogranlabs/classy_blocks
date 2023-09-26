from typing import List, Optional

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.construct.point import Point
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class DuplicatedEntry:
    """A pair vertex:{set of slave patches} that describes
    a duplicated vertex on mentioned patches"""

    def __init__(self, vertex: Vertex, patches: List[str]):
        self.vertex = vertex
        self.patches = sorted(patches)

    @property
    def point(self) -> NPPointType:
        """Vertex's point"""
        return self.vertex.position


class VertexList:
    """Handling of the 'vertices' part of blockMeshDict"""

    def __init__(self) -> None:
        self.vertices: List[Vertex] = []

        # a collection of duplicated vertices
        # belonging to a certain patch name
        self.duplicated: List[DuplicatedEntry] = []

    def find_duplicated(self, position: NPPointType, slave_patches: List[str]) -> Vertex:
        """Finds an appropriate entry in self.duplicated, if any"""
        slave_patches.sort()

        for dupe in self.duplicated:
            if f.norm(position - dupe.point) < constants.TOL:
                if dupe.patches == slave_patches:
                    return dupe.vertex

        raise VertexNotFoundError(f"No duplicated vertex found: {position} {slave_patches}")

    def find_unique(self, position: NPPointType) -> Vertex:
        """checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex"""
        for vertex in self.vertices:
            if f.norm(vertex.position - position) < constants.TOL:
                return vertex

        raise VertexNotFoundError(f"Vertex not found: {position}")

    def add(self, point: Point, slave_patches: Optional[List[str]] = None) -> Vertex:
        """Re-use existing vertices when there's already one at the position;
        unless that vertex belongs to a slave of a face-merged pair -
        in that case add a duplicate in the same position anyway"""

        # different scenarios:
        # 1. add a new vertex, nothing exist at this location yet
        # 2. reuse an existing vertex at this location
        # 3. add a new, duplicated vertex at the same location but for a different set of slave patches
        # 4. add a new, 'master' vertex because what's at the same location belongs to a slave patch

        if slave_patches is None:
            # scenario #1 and #2
            try:
                vertex = self.find_unique(point.position)

                # scenario #4:
                for dupe in self.duplicated:
                    if dupe.vertex == vertex:
                        # a point that belongs to a slave patch
                        # has been found but we need one for a 'master' patch
                        raise VertexNotFoundError
            except VertexNotFoundError:
                vertex = Vertex.from_point(point, len(self.vertices))
                self.vertices.append(vertex)

            return vertex

        # scenario 3: slave_patches is not None
        try:
            vertex = self.find_duplicated(point.position, slave_patches)
        except VertexNotFoundError:
            vertex = Vertex.from_point(point, len(self.vertices))
            self.vertices.append(vertex)
            self.duplicated.append(DuplicatedEntry(vertex, slave_patches))

        return vertex

    def clear(self) -> None:
        """Empties all lists"""
        self.vertices.clear()
        self.duplicated.clear()

    @property
    def description(self) -> str:
        """Output for blockMeshDict"""
        out = "vertices\n(\n"

        for vertex in self.vertices:
            out += f"\t{vertex.description}\n"

        out += ");\n\n"

        return out

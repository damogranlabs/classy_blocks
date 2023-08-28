from typing import List, Set

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.disk import Disk
from classy_blocks.construct.point import Point
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.items.vertex import Vertex
from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.finder import FinderBase


class RoundSolidFinder(FinderBase):
    """Find vertices on start/end faces of a round solid shape
    (Cylinder, Elbow, Frustum), ..."""

    def __init__(self, mesh: Mesh, shape: RoundSolidShape):
        super().__init__(mesh)
        self.shape = shape

    def _get_sketch(self, end_face: bool) -> Disk:
        if end_face:
            return self.shape.sketch_2

        return self.shape.sketch_1

    def _find_from_points(self, points: List[Point]) -> Set[Vertex]:
        vertices: Set[Vertex] = set()

        for point in points:
            vertices.update(self._find_by_position(point.position))

        return vertices

    def _find_from_faces(self, faces: List[Face]) -> Set[Vertex]:
        vertices: Set[Vertex] = set()

        for face in faces:
            vertices.update(self._find_from_points(face.points))

        return vertices

    def find_core(self, end_face: bool = False) -> Set[Vertex]:
        """Returns a list of vertices that define
        inner vertices of a round shape"""
        faces = self._get_sketch(end_face).core
        return self._find_from_faces(faces)

    def find_shell(self, end_face: bool = False) -> Set[Vertex]:
        """Returns a list of vertices on the
        outer edge of the shape.

        This only includes two of the vertices that define shell blocks!"""
        shell_vertices = self._find_from_faces(self._get_sketch(end_face).shell)
        core_vertices = self._find_from_faces(self._get_sketch(end_face).core)

        return shell_vertices - core_vertices

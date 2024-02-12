"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""

from classy_blocks.construct.point import Point
from classy_blocks.types import PointType
from classy_blocks.util.constants import vector_format


class Vertex(Point):
    """A 3D point in space with all transformations and an assigned index"""

    # keep the list as a class variable
    def __init__(self, position: PointType, index: int):
        super().__init__(position)

        # index in blockMeshDict; address of this object when creating edges/blocks
        self.index = index

    def __eq__(self, other):
        # When vertices are created from points,
        # it is ensured there are no duplicated at the same position.
        # Thus index is unique for the spot.
        # Same applies for __hash__.
        return self.index == other.index

    def __hash__(self):
        return self.index

    def __repr__(self):
        return f"Vertex {self.index} at {self.position}"

    @property
    def description(self) -> str:
        """Returns a string representation to be written to blockMeshDict"""
        point = vector_format(self.position)
        comment = f"// {self.index}"

        if len(self.projected_to) > 0:
            return f"project {point} ({' '.join(self.projected_to)}) {comment}"

        return f"{point} {comment}"

    @classmethod
    def from_point(cls, point: Point, index: int):
        """Creates a Vertex from point, including other properties"""
        vertex = cls(point.position, index)
        vertex.projected_to = point.projected_to

        return vertex

"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""

from typing import Optional

from classy_blocks.cbtyping import PointType
from classy_blocks.construct.point import Point


class Vertex(Point):
    """A 3D point in space with all transformations and an assigned index"""

    # keep the list as a class variable
    def __init__(self, position: PointType, index: int):
        super().__init__(position)

        # index in blockMeshDict; address of this object when creating edges/blocks
        self.index = index

        # When a vertex is duplicated for a face-merged slave patch, a new Vertex
        # is created at the same position; it keeps a reference to the original
        # (non-duplicated) vertex and the set of slave patches it was created for.
        # This is used to re-establish coincidence with non-merged neighbours while
        # still keeping the two sides of a merged interface independent.
        self.duplicated_from: Optional[Vertex] = None
        self.duplicated_patches: set[str] = set()

    @property
    def canonical_index(self) -> int:
        """Index of the underlying (non-duplicated) vertex at this position;
        duplicated vertices share the canonical index of their original."""
        if self.duplicated_from is not None:
            return self.duplicated_from.index

        return self.index

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

    @classmethod
    def from_point(cls, point: Point, index: int):
        """Creates a Vertex from point, including other properties"""
        vertex = cls(point.position, index)
        vertex.project(point.projected_to)

        return vertex

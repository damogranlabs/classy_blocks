from typing import List

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.point import Point
from classy_blocks.construct.shapes.shape import Shape


class BoundPoint:
    """A list of points at the same spot"""

    def __init__(self, point: Point):
        self.point = point

        self.faces: List[Face] = []
        self.indexes: List[int] = []

    def add(self, face: Face, index: int) -> None:
        """Adds an identifies face's point to the list of points at the same position"""
        self.faces.append(face)
        self.indexes.append(index)

    def __eq__(self, other):
        return self.point == other.point


class FusedFaceCollection:
    """Operations on a number of faces; used for creating offset shapes
    a.k.a. Hulls"""

    def __init__(self, faces: List[Face]):
        self.faces = faces
        self.bound_points: List[BoundPoint] = []

        for face in self.faces:
            for i, point in enumerate(face.points):
                self._add_point(point, face, i)

    def _add_point(self, point: Point, face: Face, index: int) -> None:
        """Get or create a new bound point, then add a face's point to it"""
        bound_point = BoundPoint(point)
        append = True

        # replace a newly created point with an existing one,
        # if found
        for existing_point in self.bound_points:
            if existing_point == bound_point:
                bound_point = existing_point
                append = False
                break

        bound_point.add(face, index)

        if append:
            self.bound_points.append(bound_point)


class Hull(Shape):
    """A Shape, created by offsetting faces.
    It will contain as many Lofts as there are faces;
    edges and projections will be dropped.

    If amount is positive, faces will be offset away
    from their parent operations and vice versa."""

    def __init__(self, faces: List[Face], amount: float):
        self.faces = FusedFaceCollection(faces)
        self.amount = amount

        # algorithm:
        # 1. identify points at the same spot
        # 2. offset points of each face normal to that face
        # 3. merge: move points from #1 to their average position
        # 4. correct amount
        # 5. create Lofts from original and offset faces

from typing import List

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.point import Point
from classy_blocks.construct.shapes.shape import Shape
from classy_blocks.types import NPPointType, NPVectorType
from classy_blocks.util import functions as f


class BoundPointError(Exception):
    """Errors with bound points"""


class BoundPointNotFoundError(BoundPointError):
    pass


class PointNotCoincidentError(BoundPointError):
    pass


class BoundPoint:
    """An index of face-points at the same spot"""

    def __init__(self, point: Point):
        self.point = point

        self.faces: List[Face] = []
        self.indexes: List[int] = []

    def add(self, face: Face, index: int) -> None:
        """Adds an identifies face's point to the list of points at the same position"""
        if face.points[index] != self.point:
            raise PointNotCoincidentError

        for i, this_face in enumerate(self.faces):
            if this_face == face:
                if index == self.indexes[i]:
                    # don't add the same face twice
                    return

        self.faces.append(face)
        self.indexes.append(index)

    @property
    def normal(self) -> NPVectorType:
        """Normal of this BoundPoint is the average normal of all touching faces"""
        normals = [f.normal for f in self.faces]

        return f.unit_vector(np.sum(normals, axis=0))

    def __eq__(self, other):
        return self.point == other.point


class BoundPointCollection:
    def __init__(self) -> None:
        self.bound_points: List[BoundPoint] = []

    def find_by_point(self, point: Point) -> BoundPoint:
        for bp in self.bound_points:
            if bp.point == point:
                return bp

        raise BoundPointNotFoundError

    def add_from_face(self, face: Face, index: int) -> BoundPoint:
        """Returns a bound point at specified location or creates a new one"""
        point = face.points[index]

        try:
            bound_point = self.find_by_point(point)
        except BoundPointNotFoundError:
            bound_point = BoundPoint(point)
            self.bound_points.append(bound_point)

        bound_point.add(face, index)

        return bound_point


class BoundFace:
    """A face that is aware of their neighbours by using bound points"""

    def __init__(self, face: Face, bound_points: List[BoundPoint]):
        self.face = face
        self.bound_points = bound_points

    def get_offset_points(self, amount: float) -> List[NPPointType]:
        """Offsets self.face in direction prescribed by bound points"""
        return [self.face.points[i].copy().translate(self.bound_points[i].normal * amount).position for i in range(4)]

    def get_offset_face(self, amount: float) -> Face:
        return Face(self.get_offset_points(amount))


class BoundFaceCollection:
    """Operations on a number of faces; used for creating offset shapes
    a.k.a. Shell"""

    def __init__(self, faces: List[Face]):
        self.faces = faces

    def get_bound_faces(self) -> List[BoundFace]:
        points_collection = BoundPointCollection()
        bound_faces: List[BoundFace] = []

        for face in self.faces:
            bound_points = [points_collection.add_from_face(face, i) for i in range(4)]
            bound_face = BoundFace(face, bound_points)
            bound_faces.append(bound_face)

        return bound_faces

    def get_offset_lofts(self, amount: float):
        # algorithm:
        # 1. identify points at the same spot
        # 2. offset points of each face normal to that face
        # 3. merge: move points from #1 to their average position
        # 4. correct amount
        # 5. create Lofts from original and offset faces

        bound_faces = self.get_bound_faces()  # No. 1
        offset_faces = [bf.get_offset_face(amount) for bf in bound_faces]  # No. 2, 3, 4

        # No. 5
        return [Loft(face, offset_faces[i]) for i, face in enumerate(self.faces)]


class Shell(Shape):
    """A Shape, created by offsetting faces.
    It will contain as many Lofts as there are faces;
    edges and projections will be dropped.

    If amount is positive, faces will be offset away
    from their parent operations and vice versa."""

    def __init__(self, faces: List[Face], amount: float):
        self.faces = faces
        self.amount = amount

        self.bound_face_collection = BoundFaceCollection(self.faces)
        self.lofts = self.bound_face_collection.get_offset_lofts(self.amount)

    @property
    def operations(self):
        return self.lofts

    def chop(self, axis: int, **kwargs) -> None:
        """Chop in offset direction"""
        for operation in self.operations:
            operation.chop(axis, **kwargs)

import functools
from typing import List

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.point import Point
from classy_blocks.construct.shapes.shape import Shape
from classy_blocks.types import NPPointType, NPVectorType
from classy_blocks.util import functions as f


class SharedPointError(Exception):
    """Errors with shared points"""


class SharedPointNotFoundError(SharedPointError):
    pass


class PointNotCoincidentError(SharedPointError):
    pass


class DisconnectedChopError(SharedPointError):
    """Issued when chopping a Shell that has disconnected faces"""


class SharedPoint:
    """A Point with knowledge of its "owner" Face(s)"""

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

    @property
    def is_shared(self) -> bool:
        """Returns False if this point is not shared
        with any other face in the collection"""
        return len(self.faces) > 1

    def __eq__(self, other):
        return self.point == other.point


class SharedPointStore:
    """A collection of shared points"""

    def __init__(self) -> None:
        self.shared_points: List[SharedPoint] = []

    def find_by_point(self, point: Point) -> SharedPoint:
        for shpoint in self.shared_points:
            if shpoint.point == point:
                return shpoint

        raise SharedPointNotFoundError

    def add_from_face(self, face: Face, index: int) -> SharedPoint:
        """Returns a shared point at specified location or creates a new one there"""
        point = face.points[index]

        try:
            shpoint = self.find_by_point(point)
        except SharedPointNotFoundError:
            shpoint = SharedPoint(point)
            self.shared_points.append(shpoint)

        shpoint.add(face, index)

        return shpoint


class AwareFace:
    """A face that is aware of their neighbours by using shared points"""

    def __init__(self, face: Face, shared_points: List[SharedPoint]):
        self.face = face
        self.shared_points = shared_points

    def get_offset_points(self, amount: float) -> List[NPPointType]:
        """Offsets self.face in direction prescribed by shared points"""
        return [self.face.points[i].copy().translate(self.shared_points[i].normal * amount).position for i in range(4)]

    def get_offset_face(self, amount: float) -> Face:
        return Face(self.get_offset_points(amount))

    @property
    def is_solitary(self) -> bool:
        """Returns True is this Face is
        not adjacent to any other face in the store"""
        return not any(sp.is_shared for sp in self.shared_points)


class AwareFaceStore:
    """Operations on a number of faces; used for creating offset shapes
    a.k.a. Shell"""

    def __init__(self, faces: List[Face]):
        self.faces = faces

    @functools.cached_property
    def point_store(self):
        points_store = SharedPointStore()

        for face in self.faces:
            for i in range(4):
                points_store.add_from_face(face, i)

        return points_store

    def get_aware_face(self, face) -> AwareFace:
        shared_points = [self.point_store.find_by_point(face.points[i]) for i in range(4)]
        return AwareFace(face, shared_points)

    @functools.cached_property
    def aware_faces(self) -> List[AwareFace]:
        aware_faces: List[AwareFace] = []

        for face in self.faces:
            aware_faces.append(self.get_aware_face(face))

        return aware_faces

    def get_offset_lofts(self, amount: float) -> List[Loft]:
        offset_faces = [awf.get_offset_face(amount) for awf in self.aware_faces]

        return [Loft(face, offset_faces[i]) for i, face in enumerate(self.faces)]

    @functools.cached_property
    def is_disconnected(self) -> bool:
        """Returns True if there are faces that are not
        connected to any other face by any point"""
        for face in self.faces:
            if self.get_aware_face(face).is_solitary:
                return True

        return False


class Shell(Shape):
    """A Shape, created by offsetting faces.
    It will contain as many Lofts as there are faces;
    edges and projections will be dropped.

    Points are offset in direction normal to their owner face;
    in case multiple faces share the same point,
    average normal is taken.

    Shell.operations will hold Lofts in the same order as
    passed faces. Use axis=2 for chopping in offset direction."""

    def __init__(self, faces: List[Face], amount: float):
        self.faces = faces
        self.amount = amount

        self.aware_face_store = AwareFaceStore(self.faces)
        self.lofts = self.aware_face_store.get_offset_lofts(self.amount)

    @property
    def operations(self):
        return self.lofts

    def chop(self, **kwargs) -> None:
        """Chop in offset direction"""
        # The lofts should be of approximately the same 'height';
        # therefore it doesn't matter which one to chop. BUT!
        # Only chop one of them because slight differences caused by
        # averaging might produce different counts.

        # issue a warning when there are disconnected lofts
        if self.aware_face_store.is_disconnected:
            raise DisconnectedChopError("There are unconnected faces in this Shell; chop its operations manually")

        self.operations[0].chop(2, **kwargs)

    def set_outer_patch(self, name: str) -> None:
        """Sets patch name for faces that have been offset"""
        for operation in self.operations:
            operation.set_patch("top", name)

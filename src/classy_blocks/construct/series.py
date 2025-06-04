from typing import Optional

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import ArrayCreationError
from classy_blocks.cbtyping import PointListType, PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE, TOL


class Series(ElementBase):
    def __init__(self, points: PointListType):
        """A list of points ('positions') in 3D space"""
        self.points = np.array(points, dtype=DTYPE)

        shape = np.shape(self.points)

        if shape[1] != 3:
            raise ArrayCreationError("Provide a list of points of 3D space!")

        if len(self.points) <= 1:
            raise ArrayCreationError("Provide at least 2 points in 3D space!")

    def translate(self, displacement):
        self.points += np.asarray(displacement, dtype=DTYPE)

        return self

    def rotate(self, angle, axis, origin: Optional[PointType] = None):
        if origin is None:
            origin = f.vector(0, 0, 0)

        axis = np.array(axis)
        matrix = f.rotation_matrix(axis, angle)
        rotated_points = np.dot(self.points - origin, matrix.T)

        self.points = rotated_points + origin

        return self

    def scale(self, ratio, origin: Optional[PointType] = None):
        if origin is None:
            origin = f.vector(0, 0, 0)

        self.points = origin + (self.points - origin) * ratio

        return self

    def mirror(self, normal: VectorType, origin: Optional[PointType] = None):
        if origin is None:
            origin = f.vector(0, 0, 0)

        normal = np.array(normal)
        matrix = f.mirror_matrix(normal)

        self.points -= origin

        mirrored_points = np.dot(self.points - origin, matrix.T)
        self.points = mirrored_points + origin

        return self

    def shear(self, normal: VectorType, origin: PointType, direction: VectorType, angle: float):
        """Move point along the plane, given by origin and normal"""
        # if the point is on the plane, do nothing
        normal = np.asarray(normal, dtype=DTYPE)
        origin = np.asarray(origin, dtype=DTYPE)

        # TODO: do this within a single array (numba?)
        for i, point in enumerate(self.points):
            if f.point_to_plane_distance(origin, normal, point) > TOL:
                distance = f.point_to_plane_distance(origin, normal, point)
                direction = f.unit_vector(direction)
                amount = distance / np.tan(angle)

                self.points[i] += direction * amount
        return self

    @property
    def center(self):
        return np.average(self.points, axis=0)

    @property
    def parts(self):
        return [self]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]

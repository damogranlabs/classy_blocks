from typing import Optional

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import ArrayCreationError
from classy_blocks.types import PointListType, PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE

# TODO! Tests


class Array(ElementBase):
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

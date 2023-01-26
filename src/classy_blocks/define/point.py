"""Defines a point in space and all operations that can be applied to it."""
from typing import Optional

import numpy as np

from classy_blocks import types
from classy_blocks.util import functions as f

class Point:
    dtype = 'float'

    """point with an index that's used in block and face definition
    and can output in OpenFOAM format"""
    def __init__(self, point:types.PointType):
        self.point = np.asarray(point, dtype=self.dtype)

        assert np.shape(self.point) == (3, ), "Provide a point in 3D space"

    def translate(self, displacement:types.VectorType):
        """Move this point by 'displacement' vector"""
        self.point += np.asarray(displacement, dtype=self.dtype)
        return self

    def rotate(self, angle:float, axis:types.VectorType, origin:types.PointType):
        """ Rotate this point around an arbitrary axis and origin """
        axis = np.asarray(axis, dtype=self.dtype)
        origin = np.asarray(origin, dtype=self.dtype)

        self.point = f.arbitrary_rotation(self.point, axis, angle, origin)
        return self

    def scale(self, ratio:float, origin:types.PointType):
        """'Scales' this point's position around given origin."""
        self.point = self.scale_point(self.point, ratio, origin)
        return self

    @staticmethod
    def scale_point(point:types.PointType, ratio:float, origin:types.PointType) -> types.PointType:
        """Scale point's position around origin."""
        point = np.asarray(point, dtype=Point.dtype)
        origin = np.asarray(origin, dtype=Point.dtype)

        return origin + (point - origin)*ratio

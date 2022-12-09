"""Vertex object defines a point in space and all operations that can be applied to it."""
from typing import Optional
from classy_blocks import types

import numpy as np
from numpy.typing import ArrayLike

from classy_blocks.util import functions as f

class Vertex:
    """point with an index that's used in block and face definition
    and can output in OpenFOAM format"""
    def __init__(self, point:types.PointType, index:int):
        self.point = np.asarray(point)
        self.index = index

    def translate(self, displacement:ArrayLike):
        """Move this point by 'displacement' vector"""
        self.point += np.asarray(displacement, dtype=float)
        return self

    def rotate(self, angle:float, axis:ArrayLike, origin:Optional[ArrayLike]=None):
        """ Rotate this vertex around an arbitrary axis and origin """
        if origin is None:
            origin = [0, 0, 0]

        # returns a new, rotated Vertex
        self.point = f.arbitrary_rotation(self.point, axis, angle, origin)
        return self

    def scale(self, ratio:float, origin:ArrayLike):
        """'Scales' this point's position around given origin."""
        self.point = self.scale_point(self.point, ratio, origin)
        return self

    @staticmethod
    def scale_point(point:ArrayLike, ratio:float, origin:ArrayLike) -> ArrayLike:
        """Scale point's position around origin."""
        return origin + (point - origin)*ratio

"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
import numpy as np

from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f


class Point:
    """A 3D point in space with all transformations"""

    def __init__(self, position:PointType):
        self.pos = np.asarray(position)
        assert np.shape(self.pos) == (3, ), "Provide a point in 3D space"

    def translate(self, displacement:VectorType):
        """Move this point by 'displacement' vector"""
        self.pos += np.asarray(displacement)
        return self

    def rotate(self, angle:float, axis:VectorType, origin:PointType):
        """ Rotate this point around an arbitrary axis and origin """
        axis = np.asarray(axis)
        origin = np.asarray(origin)

        self.pos = f.arbitrary_rotation(self.pos, axis, angle, origin)
        return self

    def scale(self, ratio:float, origin:PointType):
        """Scale point's position around origin."""
        origin = np.asarray(origin)

        self.pos = origin + (self.pos - origin)*ratio
        return self

    def __array__(self, dtype=None):
        # Enables numpy operations directly on this object
        # https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
        return np.asarray(self.pos, dtype=dtype)

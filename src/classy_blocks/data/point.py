"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
import numpy as np

from classy_blocks.types import PointType, VectorType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class Point(TransformableBase):
    """A 3D point in space with all transformations"""
    def __init__(self, position:PointType):
        self.pos = np.asarray(position)
        assert np.shape(self.pos) == (3, ), "Provide a point in 3D space"

    def translate(self, displacement:VectorType) -> 'Point':
        """Move this point by 'displacement' vector"""
        self.pos += np.asarray(displacement)
        return self

    def rotate(self, angle, axis, origin=None) -> 'Point':
        """ Rotate this point around an arbitrary axis and origin """
        axis = np.asarray(axis)

        if origin is None:
            origin = f.vector(0, 0, 0)

        origin = np.asarray(origin)

        self.pos = f.arbitrary_rotation(self.pos, f.unit_vector(axis), angle, origin)
        return self

    def scale(self, ratio, origin=None) -> 'Point':
        """Scale point's position around origin."""
        if origin is None:
            origin = f.vector(0, 0, 0)

        origin = np.asarray(origin)

        self.pos = origin + (self.pos - origin)*ratio
        return self

    def __eq__(self, other):
        return f.norm(self.pos - other.pos) < constants.tol

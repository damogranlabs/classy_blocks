"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from typing import Union, List

import numpy as np

from classy_blocks.types import PointType, VectorType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from classy_blocks.util.constants import vector_format

class Vertex:
    """A 3D point in space with all transformations and an assigned index"""
    def __init__(self, position:Union[PointType, 'Vertex']):
        self.pos = np.asarray(position)
        assert np.shape(self.pos) == (3, ), "Provide a point in 3D space"
        self.index = -1

        # TODO: project

    def translate(self, displacement:VectorType) -> 'Vertex':
        """Move this point by 'displacement' vector"""
        self.pos += np.asarray(displacement)
        return self

    def rotate(self, angle, axis, origin=None) -> 'Vertex':
        """ Rotate this point around an arbitrary axis and origin """
        axis = np.asarray(axis)

        if origin is None:
            origin = f.vector(0, 0, 0)

        origin = np.asarray(origin)

        self.pos = f.arbitrary_rotation(self.pos, f.unit_vector(axis), angle, origin)
        return self

    def scale(self, ratio, origin=None) -> 'Vertex':
        """Scale point's position around origin."""
        if origin is None:
            origin = f.vector(0, 0, 0)

        origin = np.asarray(origin)

        self.pos = origin + (self.pos - origin)*ratio
        return self

    # @property
    # def movable_entities(self) -> List['Vertex']:
    #     return [self]

    def __eq__(self, other):
        return f.norm(self.pos - other.pos) < constants.tol

    @property
    def description(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        return f"{vector_format(self.pos)} // {self.index}"

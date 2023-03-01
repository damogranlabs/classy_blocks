"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from typing import List

import numpy as np

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from classy_blocks.util.constants import vector_format

class Vertex:
    """A 3D point in space with all transformations and an assigned index"""
    # keep the list as a class variable
    registry:List['Vertex'] = []

    def __new__(cls, position:PointType, *args, **kwargs):
        """Before adding a new vertex to the list,
        check if there is already an existing one and return that instead"""
        # this is some kind of a 'singleton' pattern but stores 'globals' in a list;
        # "multipleton"
        try:
            vertex = Vertex.find(position)
            # TODO: check for face-merged stuff
        except VertexNotFoundError:
            # no vertex was found, add a new one;
            vertex = super().__new__(cls, *args, **kwargs)
            Vertex.registry.append(vertex)
        
        return vertex

    def __init__(self, position:PointType):
        if not hasattr(self, 'pos'):
            # only initialize the same Vertex once
            self.pos = np.asarray(position)
            assert np.shape(self.pos) == (3, ), "Provide a point in 3D space"

            self.index = len(self.registry) - 1

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
        return self.index == other.index

    def __repr__(self):
        return f"Vertex {self.index} at {self.pos}"

    @property
    def description(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        return f"{vector_format(self.pos)} // {self.index}"

    @staticmethod
    def find(position:PointType) -> 'Vertex':
        """checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex"""
        # TODO: optimize (octree/kdtree from scipy) (?)
        for vertex in Vertex.registry:
            if f.norm(vertex.pos - position) < constants.tol:
                return vertex

        raise VertexNotFoundError(f"Vertex not found: {str(position)}")
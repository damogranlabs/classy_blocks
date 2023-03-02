"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from typing import List, Optional

import numpy as np

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from classy_blocks.util.constants import vector_format

class Vertex(TransformableBase):
    """A 3D point in space with all transformations and an assigned index"""
    # keep the list as a class variable
    registry:List['Vertex'] = []

    pos:PointType
    index:int

    def __new__(cls, position:PointType, *args, duplicate:bool=False, **kwargs):
        """Before adding a new vertex to the list,
        check if there is already an existing one and return that instead;
        if 'copy' is True, add a new one regardless"""
        # this is some kind of a 'singleton' pattern but stores 'globals' in a list;
        # "multipleton"
        def create_new():
            new_vertex = super(Vertex, cls).__new__(cls, *args, **kwargs)
            new_vertex.index = len(Vertex.registry)

            assert np.shape(position) == (3, ), "Provide a point in 3D space"
            new_vertex.pos = np.asarray(position, dtype=constants.DTYPE)

            # TODO: project
            Vertex.registry.append(new_vertex)
            return new_vertex

        if duplicate:
            return create_new()
        try:
            return Vertex.find(position)
            # TODO: check for face-merged stuff            
        except VertexNotFoundError:
            # no vertex was found, add a new one;
            return create_new()

    def translate(self, displacement:VectorType) -> 'Vertex':
        """Move this point by 'displacement' vector"""
        self.pos += np.asarray(displacement, dtype=constants.DTYPE)
        return self

    def rotate(self, angle, axis, origin=None) -> 'Vertex':
        """ Rotate this point around an arbitrary axis and origin """
        self.pos = f.rotate(self.pos, f.unit_vector(axis), angle, origin)
        return self

    def scale(self, ratio:float, origin:Optional[PointType]=None) -> 'Vertex':
        """Scale point's position around origin."""
        self.pos = f.scale(self.pos, ratio, origin)
        return self

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
        position = np.asarray(position, dtype=constants.DTYPE)

        for vertex in Vertex.registry:
            if f.norm(vertex.pos - position) < constants.TOL:
                return vertex

        raise VertexNotFoundError(f"Vertex not found: {str(position)}")

    # @staticmethod
    # def copy(position:PointType) -> 'Vertex':
    #     """Returns a new Vertex regardless of whether
    #     an existing one is already at given position"""
    #     try:
    #         existing_vertex = Vertex.find(position)
    #         new_vertex = copy.copy(existing_vertex)
    #         new_vertex.index = len(Vertex.registry)
    #         Vertex.registry.append(new_vertex)
    #         return new_vertex
    #     except VertexNotFoundError:
    #         # make a new one
    #         return Vertex(position)

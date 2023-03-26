"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from typing import Optional

import numpy as np

from classy_blocks.base.transformable import TransformableBase
from classy_blocks.types import PointType, VectorType, ProjectToType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from classy_blocks.util.constants import vector_format

class Vertex(TransformableBase):
    """A 3D point in space with all transformations and an assigned index"""
    # keep the list as a class variable
    def __init__(self, position:PointType, index:int):
        position = np.asarray(position, dtype=constants.DTYPE)
        assert np.shape(position) == (3, ), "Provide a point in 3D space"
        self.pos = position

        # index in blockMeshDict; address of this object when creating edges/blocks
        self.index = index

        self.project_to:Optional[ProjectToType] = None

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

    def project(self, geometry:ProjectToType) -> None:
        """Project this vertex to a single or multiple geometries"""
        if not isinstance(geometry, list):
            geometry = [geometry]

        self.project_to = geometry

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return f"Vertex {self.index} at {self.pos}"

    @property
    def description(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        point = vector_format(self.pos)
        comment = f"// {self.index}"

        if self.project_to is not None:
            return f"project {point} ({' '.join(self.project_to)}) {comment}"

        return f"{point} {comment}"

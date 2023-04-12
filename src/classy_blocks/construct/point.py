from typing import TypeVar

import numpy as np

from classy_blocks.types import PointType, ProjectToType
from classy_blocks.base.element import ElementBase
from classy_blocks.util.constants import DTYPE, TOL, vector_format
from classy_blocks.util import functions as f

PointT = TypeVar("PointT", bound="Point")


class Point(ElementBase):
    """A 3D point in space with optional projection
    to a set of surfaces and transformations"""

    def __init__(self, position: PointType):
        self.pos = np.asarray(position, dtype=DTYPE)
        assert np.shape(self.pos) == (3,), "Provide a point in 3D space"

        self.project_to: ProjectToType = []

    def translate(self, displacement):
        """Move this point by 'displacement' vector"""
        self.pos += np.asarray(displacement, dtype=DTYPE)
        return self

    def rotate(self, angle, axis, origin=None):
        """Rotate this point around an arbitrary axis and origin"""
        self.pos = f.rotate(self.pos, angle, f.unit_vector(axis), origin)
        return self

    def scale(self, ratio, origin=None):
        """Scale point's position around origin."""
        self.pos = f.scale(self.pos, ratio, origin)
        return self

    def project(self, geometry: ProjectToType) -> None:
        """Project this vertex to a single or multiple geometries"""
        if not isinstance(geometry, list):
            geometry = [geometry]

        self.project_to = geometry

    def __eq__(self, other):
        return f.norm(self.pos - other.pos) < TOL

    @property
    def description(self) -> str:
        """Returns a string representation to be written to blockMeshDict"""
        return vector_format(self.pos)

    @property
    def components(self):
        return [self]


# An 'alias' to avoid confusion in mathematical lingo
Vector = Point

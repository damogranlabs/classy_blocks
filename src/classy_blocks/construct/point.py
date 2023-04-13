from typing import TypeVar

import numpy as np

from classy_blocks.types import PointType, ProjectToType, NPVectorType
from classy_blocks.base.element import ElementBase
from classy_blocks.util.constants import DTYPE, TOL, vector_format
from classy_blocks.util import functions as f

PointT = TypeVar("PointT", bound="Point")


class Point(ElementBase):
    """A 3D point in space with optional projection
    to a set of surfaces and transformations"""

    def __init__(self, position: PointType):
        self.position = np.asarray(position, dtype=DTYPE)
        assert np.shape(self.position) == (3,), "Provide a point in 3D space"

        self.projected_to: ProjectToType = []

    def translate(self, displacement):
        """Move this point by 'displacement' vector"""
        self.position += np.asarray(displacement, dtype=DTYPE)
        return self

    def rotate(self, angle, axis, origin=None):
        """Rotate this point around an arbitrary axis and origin"""
        self.position = f.rotate(self.position, angle, f.unit_vector(axis), origin)
        return self

    def scale(self, ratio, origin=None):
        """Scale point's position around origin."""
        self.position = f.scale(self.position, ratio, origin)
        return self

    def project(self, geometry: ProjectToType) -> None:
        """Project this vertex to a single or multiple geometries"""
        if not isinstance(geometry, list):
            geometry = [geometry]

        self.projected_to = geometry

    @property
    def description(self) -> str:
        """Returns a string representation to be written to blockMeshDict"""
        return vector_format(self.position)

    @property
    def parts(self):
        return [self]

    def __eq__(self, other):
        return f.norm(self.position - other.pos) < TOL

    def __repr__(self):
        return f"Point {self.description}"


class Vector(Point):
    """An 'alias' to avoid confusion in mathematical lingo"""

    @property
    def components(self) -> NPVectorType:
        """Vector's components (same as point's position but more
        grammatically/mathematically accurate)"""
        return self.position

    def __repr__(self):
        return f"Vector {self.description}"

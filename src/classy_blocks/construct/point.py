from typing import Optional, TypeVar

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import PointCreationError
from classy_blocks.cbtyping import NPVectorType, PointType, ProjectToType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE, TOL, vector_format

PointT = TypeVar("PointT", bound="Point")


class Point(ElementBase):
    """A 3D point in space with optional projection
    to a set of surfaces and transformations"""

    def __init__(self, position: PointType):
        self.position = np.array(position, dtype=DTYPE)
        if not np.shape(self.position) == (3,):
            raise PointCreationError("Provide a point in 3D space", f"Position: {position}")

        self._projected_to: set[str] = set()

    def move_to(self, position: PointType) -> None:
        """Move this point to supplied position"""
        self.position[0] = position[0]
        self.position[1] = position[1]
        self.position[2] = position[2]

    def translate(self, displacement):
        """Move this point by 'displacement' vector"""
        self.position += np.asarray(displacement, dtype=DTYPE)
        return self

    def rotate(self, angle, axis, origin: Optional[PointType] = None):
        """Rotate this point around an arbitrary axis and origin"""
        if origin is None:
            origin = f.vector(0, 0, 0)

        self.position = f.rotate(self.position, angle, f.unit_vector(axis), origin)
        return self

    def scale(self, ratio, origin: Optional[PointType] = None):
        """Scale point's position around origin."""
        if origin is None:
            origin = f.vector(0, 0, 0)

        self.position = f.scale(self.position, ratio, origin)
        return self

    def mirror(self, normal: VectorType, origin: Optional[PointType] = None):
        """Mirror (reflect) the point around a plane, defined by normal vector and a passing point"""
        if origin is None:
            origin = f.vector(0, 0, 0)

        self.position = f.mirror(self.position, normal, origin)
        return self

    def shear(self, normal: VectorType, origin: PointType, direction: VectorType, angle: float):
        """Move point along the plane, given by origin and normal"""
        # if the point is on the plane, do nothing
        normal = np.asarray(normal, dtype=DTYPE)
        origin = np.asarray(origin, dtype=DTYPE)

        distance = f.point_to_plane_distance(origin, normal, self.position)

        if distance > TOL:
            direction = f.unit_vector(direction)
            amount = distance / np.tan(angle)

            self.position += direction * amount
        return self

    def project(self, label: ProjectToType) -> None:
        """Project this vertex to a single or multiple geometries"""
        if not isinstance(label, list):
            label = [label]

        self._projected_to.update(label)

    @property
    def parts(self):
        return [self]

    @property
    def center(self):
        return self.position

    @property
    def projected_to(self) -> list[str]:
        return sorted(list(self._projected_to))

    def __eq__(self, other):
        return f.norm(self.position - other.position) < TOL

    def __repr__(self):
        return f"Point {vector_format(self.position)}"

    def __str__(self):
        return repr(self)


class Vector(Point):
    """An 'alias' to avoid confusion in mathematical lingo"""

    @property
    def components(self) -> NPVectorType:
        """Vector's components (same as point's position but more
        grammatically/mathematically accurate)"""
        return self.position

    def __repr__(self):
        return f"Vector {vector_format(self.position)}"

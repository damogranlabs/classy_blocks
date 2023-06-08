from typing import List

import numpy as np

from classy_blocks.base.exceptions import AnnulusCreationError
from classy_blocks.construct.edges import Origin
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.sketch import Sketch
from classy_blocks.types import NPPointType, PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class Annulus(Sketch):
    """A base for ring-like shapes;
    In real-life, Annulus and Ring are the same 2D objects.
    Here, however, Annulus is a 2D collection of faces whereas
    Ring is an annulus that has been extruded to 3D."""

    def __init__(
        self,
        center_point: PointType,
        outer_radius_point: PointType,
        normal: VectorType,
        inner_radius: float,
        n_segments: int = 8,
    ):
        center_point = np.asarray(center_point)
        normal = f.unit_vector(np.asarray(normal))
        outer_radius_point = np.asarray(outer_radius_point)
        inner_radius_point = center_point + f.unit_vector(outer_radius_point - center_point) * inner_radius
        segment_angle = 2 * np.pi / n_segments

        face = Face(
            [  # points
                inner_radius_point,
                outer_radius_point,
                f.rotate(outer_radius_point, segment_angle, normal, center_point),
                f.rotate(inner_radius_point, segment_angle, normal, center_point),
            ],
            [None, Origin(center_point), None, Origin(center_point)],  # edges
            check_coplanar=True,
        )

        self.core: List[Face] = []
        self.shell = [face.copy().rotate(i * segment_angle, normal, center_point) for i in range(n_segments)]

        if self.inner_radius > self.outer_radius:
            raise AnnulusCreationError(
                "Outer ring radius must be larger than inner!",
                f"Inner radius: {self.inner_radius}, Outer radius: {self.outer_radius}",
            )
        diff = abs(np.dot(normal, outer_radius_point - center_point))
        if diff > TOL:
            raise AnnulusCreationError(
                "Normal and radius are not perpendicular!", f"Difference: {diff}, tolerance: {TOL}"
            )

    @property
    def faces(self) -> List[Face]:
        return self.shell

    @property
    def center(self) -> NPPointType:
        return np.average([f.center for f in self.faces], axis=0)

    @property
    def n_segments(self):
        return len(self.faces)

    # FIXME: do something with inner_/outer/_point/_vector confusion
    @property
    def outer_radius_point(self) -> NPPointType:
        """Point at outer radius, 0-th segment"""
        return self.faces[0].point_array[1]

    @property
    def outer_radius(self) -> float:
        """Outer radius"""
        return f.norm(self.outer_radius_point - self.center)

    @property
    def radius(self) -> float:
        """Outer radius"""
        return self.outer_radius

    @property
    def radius_point(self) -> NPPointType:
        """See self.outer_radius_point"""
        return self.outer_radius_point

    @property
    def inner_radius_point(self) -> NPPointType:
        """Point at inner radius, 0-th segment"""
        return self.faces[0].point_array[0]

    @property
    def inner_radius(self) -> float:
        """Returns inner radius as length, that is, distance between
        center and inner radius point"""
        return f.norm(self.inner_radius_point - self.center)

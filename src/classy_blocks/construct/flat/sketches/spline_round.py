from typing import ClassVar, List, Optional

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.edges import Origin, Spline
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.disk import DiskBase
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.point import Point
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class SplineRound(MappedSketch):
    """
    Base class for spline round sketches.
    Shape can be oval, elliptical or circular.
    """

    spline_ratios = np.asarray(DiskBase.spline_ratios)
    core_ratio = DiskBase.core_ratio

    n_outer_spline_points = 20

    chops: ClassVar = [
        [1],  # axis 0
        [1, 2],  # axis 1
    ]

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
    ):
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        note the vectors from the center to corner 1 and 2 should be perpendicular.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and eliptical shape
            corner_2_point: Radius for circular and eliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
        """
        self.center = np.asarray(center_point)
        self.corner_1 = np.asarray(corner_1_point)
        self.corner_2 = np.asarray(corner_2_point)

        self.side_1 = float(side_1)
        self.side_2 = float(side_2)

    @property
    def center(self) -> PointType:
        """
        Returns center point defined as center_point in __init__ in stable way after transforms.

        Ensure correct face and point is used when subclassing!
        """
        try:
            return np.asarray(self.faces[0].points[3].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._center

    @center.setter
    def center(self, center_point: PointType):
        """
        Setter method for center point.
        Used prior initialization of MappedSketch.
        """
        self._center = np.asarray(center_point)

    @property
    def corner_1(self) -> PointType:
        """
        Returns corner 1 defined as corner_1_point in __init__ in stable way after transforms.

        Ensure correct face and point is used when subclassing!
        """
        try:
            return np.asarray(self.faces[2].points[1].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_1

    @corner_1.setter
    def corner_1(self, corner_1_point: PointType):
        """
        Setter method for corner point.
        Used prior initialization of MappedSketch.
        """
        self._corner_1 = np.asarray(corner_1_point)

    @property
    def corner_2(self) -> PointType:
        """
        Returns corner 2 defined as corner_2_point in __init__ in stable way after transforms.

        Ensure correct face and point is used when subclassing!
        """
        try:
            return np.asarray(self.faces[1].points[2].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_2

    @corner_2.setter
    def corner_2(self, corner_2_point: PointType):
        """
        Setter method for corner point.
        Used prior initialization of MappedSketch.
        """
        self._corner_2 = np.asarray(corner_2_point)

    @property
    def r_1(self) -> float:
        """Returns radius 1 in stable way after transforms."""
        return f.norm(self.corner_1 - self.center) - self.side_1

    @property
    def r_2(self) -> float:
        """Returns radius 2 in stable way after transforms."""
        return f.norm(self.corner_2 - self.center) - self.side_2

    @property
    def u_0(self) -> VectorType:
        """Returns unit vector 0 in stable way after transforms."""
        return f.unit_vector(np.cross(self.u_1, self.u_2))

    @property
    def normal(self) -> VectorType:
        return self.u_0

    @property
    def u_1(self) -> VectorType:
        """Returns unit vector 1 in stable way after transforms."""
        return f.unit_vector(self.corner_1 - self.center)

    @property
    def u_2(self) -> VectorType:
        """Returns unit vector 2 in stable way after transforms."""
        return f.unit_vector(self.corner_2 - self.center)

    def scale(self: ElementBase, ratio: float, origin: Optional[PointType] = None) -> ElementBase:
        """Reimplementation of scale to include side_1 and side_2."""

        self.side_1 = ratio * self.side_1
        self.side_2 = ratio * self.side_2

        return super().scale(ratio, Origin)

    @classmethod
    def init_from_radius(cls, center_point, corner_1_point, corner_2_point, r_1, r_2):
        """Calcluate the side lengths based on the radius and return sketch"""
        side_1 = f.norm(corner_1_point - center_point) - r_1
        side_2 = f.norm(corner_2_point - center_point) - r_2

        return cls(center_point, corner_1_point, corner_2_point, side_1, side_2)


class QuarterSplineRound(SplineRound):
    """Sketch for Quater oval, eliptical and circular shapes"""

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
    ) -> None:
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        note the vectors from the center to corner 1 and 2 should be perpendicular.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and eliptical shape
            corner_2_point: Radius for circular and eliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
        """
        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2)
        # Points
        p0 = self.center
        p1 = self.center + (self.side_2 + self.core_ratio * self.r_2) * self.u_2
        p2 = self.corner_2
        p3 = self.center + (self.side_1 + self.core_ratio * self.r_1) * self.u_1
        p4 = (
            self.center
            + (self.side_1 + self.spline_ratios[7] * self.r_1) * self.u_1
            + (self.side_2 + self.spline_ratios[7] * self.r_2) * self.u_2
        )
        p5 = self.corner_1
        p6 = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.r_2) * self.u_2
        )

        quad_map = [
            # core
            (3, 4, 1, 0),
            # Shell
            (4, 6, 2, 1),
            (3, 5, 6, 4),
        ]

        positions = [p0, p1, p2, p3, p4, p5, p6]
        super(SplineRound, self).__init__(positions, quad_map)

    @SplineRound.center.getter
    def center(self) -> PointType:
        """
        Returns center point defined as center_point in __init__ in stable way after transforms.
        """
        try:
            return np.asarray(self.faces[0].points[3].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._center

    @SplineRound.corner_1.getter
    def corner_1(self) -> PointType:
        """
        Returns corner 1 defined as corner_1_point in __init__ in stable way after transforms.
        """
        try:
            return np.asarray(self.faces[2].points[1].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_1

    @SplineRound.corner_2.getter
    def corner_2(self) -> PointType:
        """
        Returns corner 2 defined as corner_2_point in __init__ in stable way after transforms.
        """
        try:
            return np.asarray(self.faces[1].points[2].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_2

    @property
    def grid(self) -> List[List[Face]]:
        return [[self.faces[0]], self.faces[1:]]

    @property
    def core(self) -> List[Face]:
        return self.grid[0]

    @property
    def shell(self) -> List[Face]:
        return self.grid[-1]

    def add_edges(self) -> None:
        # Shell 1
        core_spline_1_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.spline_ratios[-1:-8:-1].reshape((-1, 1)) / 0.8 * self.core_ratio * self.r_2 * self.u_2
            + self.spline_ratios[:7].reshape((-1, 1)) / 0.8 * self.core_ratio * self.r_1 * self.u_1
        )
        theta = np.linspace(0, np.pi / 4, self.n_outer_spline_points + 1, endpoint=False)[1:].reshape((-1, 1))
        shell_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2 * np.cos(theta) * self.u_2
            + self.r_1 * np.sin(theta) * self.u_1
        )

        # If oval shape 3 spline points are added to ensure a straight line
        if self.side_1 > constants.TOL:
            core_side_points = np.linspace(
                self.center + (self.side_2 + self.core_ratio * self.r_2) * self.u_2 + 0.95 * self.side_1 * self.u_1,
                self.center + (self.side_2 + self.core_ratio * self.r_2) * self.u_2 + self.side_1 * self.u_1,
                3,
            )
            core_spline_1_points = np.insert(core_spline_1_points, 0, core_side_points, axis=0)

            shell_side_points = np.linspace(
                self.center + (self.side_2 + self.r_2) * self.u_2 + 0.95 * self.side_1 * self.u_1,
                self.center + (self.side_2 + self.r_2) * self.u_2 + self.side_1 * self.u_1,
                3,
            )
            shell_curve_points = np.insert(shell_curve_points, 0, shell_side_points, axis=0)

        # Add edges to shell 1
        self.shell[0].add_edge(1, Spline(shell_curve_points[::-1]))
        self.core[0].add_edge(1, Spline(core_spline_1_points[::-1]))

        # Shell 2
        core_spline_2_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.spline_ratios[7:].reshape((-1, 1)) / 0.8 * self.core_ratio * self.r_1 * self.u_1
            + self.spline_ratios[6::-1].reshape((-1, 1)) / 0.8 * self.core_ratio * self.r_2 * self.u_2
        )

        theta = np.linspace(np.pi / 4, np.pi / 2, self.n_outer_spline_points + 1, endpoint=False)[1:].reshape((-1, 1))
        shell_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2 * np.cos(theta) * self.u_2
            + self.r_1 * np.sin(theta) * self.u_1
        )
        # If oval shape 3 spline points are added to ensure a straight line
        if self.side_2 > constants.TOL:
            core_side_points = np.linspace(
                self.center + 0.95 * self.side_2 * self.u_2 + (self.side_1 + self.core_ratio * self.r_1) * self.u_1,
                self.center + self.side_2 * self.u_2 + (self.side_1 + self.core_ratio * self.r_1) * self.u_1,
                3,
            )
            core_spline_2_points = np.append(core_spline_2_points, core_side_points[::-1], axis=0)

            shell_side_points = np.linspace(
                self.center + (self.side_1 + self.r_1) * self.u_1 + 0.95 * self.side_2 * self.u_2,
                self.center + (self.side_1 + self.r_1) * self.u_1 + self.side_2 * self.u_2,
                3,
            )

            shell_curve_points = np.append(shell_curve_points, shell_side_points[::-1], axis=0)

        # Add edges to shell 1
        self.shell[1].add_edge(1, Spline(shell_curve_points[::-1]))
        self.core[0].add_edge(0, Spline(core_spline_2_points[::-1]))

        # If a circular shape use arc instead of spline
        if self.side_1 <= constants.TOL and self.side_2 <= constants.TOL and abs(self.r_1 - self.r_2) < constants.TOL:
            self.shell[0].add_edge(1, Origin(self.center))
            self.shell[1].add_edge(1, Origin(self.center))


class QuarterSplineRoundRing(SplineRound):
    """Ring based on SplineRound."""

    chops: ClassVar = [
        [0],  # axis 0
        [0, 1],  # axis 1
    ]

    def __init__(self, center_point, corner_1_point, corner_2_point, side_1, side_2, width_1, width_2):
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        Note the vectors from the center to corner 1 and 2 should be perpendicular.
        The ring is defined such it will fit around a QuaterSplineRound defined with the same center, corners and sides.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and eliptical shape
            corner_2_point: Radius for circular and eliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
            width_1: Width of shell
            width_2: Width of shell
        """
        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2)
        self.width_1 = float(width_1)
        self.width_2 = float(width_2)

        p2 = self.corner_2
        p2_2 = self.corner_2 + self.width_2 * self.u_2
        p5 = self.corner_1
        p5_2 = self.corner_1 + self.width_1 * self.u_1
        p6 = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.r_2) * self.u_2
        )
        p6_2 = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.r_1_outer) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.r_2_outer) * self.u_2
        )

        quad_map = [(2, 3, 1, 0), (4, 5, 3, 2)]

        positions = [p2, p2_2, p6, p6_2, p5, p5_2]
        super(SplineRound, self).__init__(positions, quad_map)

    @property
    def center(self) -> PointType:
        """
        Returns center point defined as center_point in __init__.
        Note for this to be stable it has to be handled under transformation.
        """
        return self._center.position

    @center.setter
    def center(self, center_point: PointType):
        self._center = Point(center_point)

    @SplineRound.corner_1.getter
    def corner_1(self) -> PointType:
        """
        Returns corner 1 defined as corner_1_point in __init__ in stable way after transforms.
        """
        try:
            return np.asarray(self.faces[1].points[0].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_1

    @SplineRound.corner_2.getter
    def corner_2(self) -> PointType:
        """
        Returns corner 2 defined as corner_2_point in __init__ in stable way after transforms.
        """
        try:
            return np.asarray(self.faces[0].points[3].position)
        # Fall back to return value set in __init__
        except AttributeError:
            return self._corner_2

    @property
    def r_1_outer(self) -> float:
        """Returns radius 1 in stable way after transforms."""
        return f.norm(self.corner_1 - self.center) - self.side_1 + self.width_1

    @property
    def r_2_outer(self) -> float:
        """Returns radius 2 in stable way after transforms."""
        return f.norm(self.corner_2 - self.center) - self.side_2 + self.width_2

    @property
    def grid(self) -> List[Face]:
        return [self.faces]

    @property
    def core(self) -> List[Face]:
        return self.grid[0]

    @property
    def shell(self) -> List[Face]:
        return self.grid[-1]

    @property
    def parts(self):
        return [*super(SplineRound, self).parts, self._center]

    def scale(self: ElementBase, ratio: float, origin: Optional[PointType] = None) -> ElementBase:
        """Reimplementation of scale to include side_1 and side_2."""

        self.side_1 = ratio * self.side_1
        self.side_2 = ratio * self.side_2

        self.width_1 = ratio * self.width_1
        self.width_2 = ratio * self.width_2

        return super().scale(ratio, Origin)

    def add_edges(self) -> None:
        # Shell 1
        theta = np.linspace(0, np.pi / 4, self.n_outer_spline_points + 1, endpoint=False)[1:].reshape((-1, 1))
        shell_inner_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2 * np.cos(theta) * self.u_2
            + self.r_1 * np.sin(theta) * self.u_1
        )
        shell_outer_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2_outer * np.cos(theta) * self.u_2
            + self.r_1_outer * np.sin(theta) * self.u_1
        )

        # If oval shape 3 spline points are added to ensure a straight line
        if self.side_1 > constants.TOL:
            shell_inner_side_points = np.linspace(
                self.center + (self.side_2 + self.r_2) * self.u_2 + 0.95 * self.side_1 * self.u_1,
                self.center + (self.side_2 + self.r_2) * self.u_2 + self.side_1 * self.u_1,
                3,
            )
            shell_inner_curve_points = np.insert(shell_inner_curve_points, 0, shell_inner_side_points, axis=0)

            shell_outer_side_points = np.linspace(
                self.center + (self.side_2 + self.r_2_outer) * self.u_2 + 0.95 * self.side_1 * self.u_1,
                self.center + (self.side_2 + self.r_2_outer) * self.u_2 + self.side_1 * self.u_1,
                3,
            )
            shell_outer_curve_points = np.insert(shell_outer_curve_points, 0, shell_outer_side_points, axis=0)

        # Add edges to shell 1
        self.shell[0].add_edge(3, Spline(shell_inner_curve_points[::-1]))
        self.shell[0].add_edge(1, Spline(shell_outer_curve_points[::-1]))

        # Shell 2
        theta = np.linspace(np.pi / 4, np.pi / 2, 10, endpoint=False)[1:].reshape((-1, 1))
        shell_inner_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2 * np.cos(theta) * self.u_2
            + self.r_1 * np.sin(theta) * self.u_1
        )
        shell_outer_curve_points = (
            self.center
            + self.side_2 * self.u_2
            + self.side_1 * self.u_1
            + self.r_2_outer * np.cos(theta) * self.u_2
            + self.r_1_outer * np.sin(theta) * self.u_1
        )

        # If oval shape 3 spline points are added to ensure a straight line
        if self.side_2 > constants.TOL:
            shell_inner_side_points = np.linspace(
                self.center + (self.side_1 + self.r_1) * self.u_1 + 0.95 * self.side_2 * self.u_2,
                self.center + (self.side_1 + self.r_1) * self.u_1 + self.side_2 * self.u_2,
                3,
            )
            shell_inner_curve_points = np.append(shell_inner_curve_points, shell_inner_side_points[::-1], axis=0)

            shell_outer_side_points = np.linspace(
                self.center + (self.side_1 + self.r_1_outer) * self.u_1 + 0.95 * self.side_2 * self.u_2,
                self.center + (self.side_1 + self.r_1_outer) * self.u_1 + self.side_2 * self.u_2,
                3,
            )
            shell_outer_curve_points = np.append(shell_outer_curve_points, shell_outer_side_points[::-1], axis=0)

        # Add edges to shell 2
        self.shell[1].add_edge(3, Spline(shell_inner_curve_points[::-1]))
        self.shell[1].add_edge(1, Spline(shell_outer_curve_points[::-1]))

        # If a circular shape use arc instead of spline
        if self.side_1 <= constants.TOL and self.side_2 <= constants.TOL and abs(self.r_1 - self.r_2) < constants.TOL:
            self.shell[0].add_edge(1, Origin(self.center))
            self.shell[1].add_edge(1, Origin(self.center))
            self.shell[0].add_edge(3, Origin(self.center))
            self.shell[1].add_edge(3, Origin(self.center))

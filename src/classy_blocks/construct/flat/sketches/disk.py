import abc
from typing import ClassVar, Optional

import numpy as np

from classy_blocks.cbtyping import (
    IndexType,
    NPPointListType,
    NPPointType,
    NPVectorType,
    PointListType,
    PointType,
    VectorType,
)
from classy_blocks.construct.edges import Origin, Spline
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.point import Point
from classy_blocks.util import functions as f


class FanPattern:
    """A helper class for calculation of cylinder points"""

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        self.center_point = np.asarray(center_point)
        self.normal = f.unit_vector(np.asarray(normal))
        self.radius_point = np.asarray(radius_point)

        self.radius_vector = self.radius_point - self.center_point

    def get_outer_points(self, angles) -> NPPointListType:
        return np.array([f.rotate(self.radius_point, a, self.normal, self.center_point) for a in angles])

    def get_inner_points(self, angles, ratios: list[float]) -> NPPointListType:
        """Inner points are scaled back by defined ratios
        that repeat over the circumference"""
        points = self.get_outer_points(angles)

        for i, point in enumerate(points):
            ratio = ratios[i % len(ratios)]
            points[i] = self.center_point + (point - self.center_point) * ratio

        return points


class DiskBase(MappedSketch, abc.ABC):
    # Ratios between core and outer points:
    # Relative size of lines 0-1 and 0-4 (in QuarterDisk, others analogously):
    # Just the right value will yield the lowest non-orthogonality and skewness;
    # determined empirically
    core_ratio = 0.8

    # Spline points for optimized round meshes
    # As ratio of radius
    spline_ratios = (
        0.112269,
        0.222463,
        0.326692,
        0.421177,
        0.502076,
        0.566526,
        0.610532,
        0.610535,
        0.625913,
        0.652300,
        0.686516,
        0.724996,
        0.762587,
        0.792025,
    )

    def __init__(self, positions: PointListType, quads: list[IndexType]):
        # Center point as a constant.
        self.origo_point = Point(positions[0])

        super().__init__(positions, quads)

    @property
    def origo(self):
        return self.origo_point.position

    # Relative size of the inner square (O-1-2-3) in a single core cylinder:
    # - too small will cause unnecessary high number of small cells in the square;
    # - too large will prevent creating large numbers of boundary layers
    @property
    def diagonal_ratio(self) -> float:
        return 2**0.5 * self.spline_ratios[7] / 0.8 * self.core_ratio

    def circular_core_spline(
        self,
        p_core_ratio: PointType,
        p_diagonal_ratio: PointType,
        reverse: bool = False,
        center: Optional[PointType] = None,
    ) -> NPPointListType:
        """Creates the spline points for the core."""
        p_0 = np.asarray(p_core_ratio)
        p_1 = np.asarray(p_diagonal_ratio)
        if center is None:
            center = self.center

        # Spline points in unitary coordinates
        spline_points_u = np.array([self.spline_ratios[-1:6:-1]]).T * np.array([0, 1, 0]) + np.array(
            [self.spline_ratios[:7]]
        ).T * np.array([0, 0, 1])

        # p_1 and p_2 in unitary coordinates
        p_0_u = np.array([0, 0.8, 0])
        p_1_u = np.array([0, 6.10535e-01, 6.10535e-01])

        # orthogonal vectors based on p_0_u and p_1_u
        u_0_org = p_0_u
        u_1_org = p_1_u - np.dot(p_1_u, f.unit_vector(u_0_org)) * f.unit_vector(u_0_org)

        # Spline points in u_0_org and u_1_org
        spline_d_0_org = np.dot(spline_points_u, f.unit_vector(u_0_org)).reshape((-1, 1)) / f.norm(u_0_org)
        spline_d_1_org = np.dot(spline_points_u, f.unit_vector(u_1_org)).reshape((-1, 1)) / f.norm(u_1_org)

        # New plane defined by new points
        u_0 = p_0 - center
        u_1 = p_1 - center - np.dot(p_1 - center, f.unit_vector(u_0)) * f.unit_vector(u_0)

        spline_points_new = center + spline_d_0_org * u_0 + spline_d_1_org * u_1
        if reverse:
            return spline_points_new[::-1]
        else:
            return spline_points_new

    def add_core_spline_edges(self) -> None:
        """Add a spline to the core blocks for an optimized mesh."""
        for i, face in enumerate(self.core):
            p_0 = face.point_array[(i + 1) % 4]  # Core point on radius vector
            p_1 = face.point_array[(i + 2) % 4]  # Core point on diagonal
            p_2 = face.point_array[(i + 3) % 4]  # Core point on perpendicular radius vector

            spline_curve_0_1 = Spline(self.circular_core_spline(p_0, p_1, reverse=i == 2))
            spline_curve_1_2 = Spline(self.circular_core_spline(p_2, p_1, reverse=i != 1))

            # Add curves to edges
            edge_1 = (i + 1) % 4
            edge_2 = (i + 2) % 4
            face.add_edge(edge_1, spline_curve_0_1)
            face.add_edge(edge_2, spline_curve_1_2)

    def add_edges(self):
        for face in self.shell:
            face.add_edge(1, Origin(self.origo))

        self.add_core_spline_edges()

    @property
    def center(self) -> NPPointType:
        """Center point of this sketch"""
        return self.faces[0].points[0].position

    @property
    def radius_point(self) -> NPPointType:
        """Point at outer radius"""
        return self.shell[0].points[1].position

    @property
    def radius_vector(self) -> NPVectorType:
        """Vector that points from center of this
        *Circle to its (first) radius point
        Origo is used instead of center to ensure outside is constant, when moving the core,
        does not change the outer shape."""
        return self.radius_point - self.origo

    @property
    def radius(self) -> float:
        """Radius of this *circle, length of self.radius_vector"""
        return float(f.norm(self.radius_vector))

    @property
    def n_segments(self):
        return len(self.grid[1])

    @property
    def core(self) -> list[Face]:
        return self.grid[0]

    @property
    def shell(self) -> list[Face]:
        return self.grid[-1]

    @property
    def parts(self):
        return [self.origo_point, *super().parts]


class OneCoreDisk(DiskBase):
    """A disk with a single block in  the center and four blocks around;
    see docs/sketches for point numbers and faces/grid indexing."""

    chops: ClassVar = [
        [1],  # axis 0
        [1, 2],  # axis 1
    ]

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        quad_map = [
            # core
            [0, 1, 2, 3],
            # shell
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ]

        pattern = FanPattern(center_point, radius_point, normal)
        ratios = [self.diagonal_ratio]
        angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)

        super().__init__([*pattern.get_inner_points(angles, ratios), *pattern.get_outer_points(angles)], quad_map)

        # correct origo_point as it is not the same as with FourCoreDisk-based sketches
        self.origo_point = Point(center_point)

    @property
    def center(self):
        return self.faces[0].center

    @property
    def grid(self):
        return [self.faces[:1], self.faces[1:]]

    def add_core_spline_edges(self):
        pass


class QuarterDisk(DiskBase):
    """A quarter of a four-core disk; see docs/sketches for point numbers and faces/grid indexing"""

    chops: ClassVar = [
        [1],  # axis 0
        [1, 2],  # axis 1
    ]

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        quad_map = [
            # core
            [0, 1, 2, 3],
            # shell
            [1, 4, 5, 2],
            [2, 5, 6, 3],
        ]

        pattern = FanPattern(center_point, radius_point, normal)
        ratios = [self.core_ratio, self.diagonal_ratio]
        angles = np.linspace(0, np.pi / 2, num=3)

        super().__init__(
            [center_point, *pattern.get_inner_points(angles, ratios), *pattern.get_outer_points(angles)], quad_map
        )

    @property
    def grid(self):
        return [[self.faces[0]], self.faces[1:]]


class HalfDisk(DiskBase):
    """One half of a four-core disk"""

    chops: ClassVar = [
        [2],  # axis 0
        [2, 3, 4],  # axis 1
    ]

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        quad_map = [
            # core
            [0, 1, 2, 3],
            [5, 0, 3, 4],
            # shell
            [1, 6, 7, 2],
            [2, 7, 8, 3],
            [3, 8, 9, 4],
            [4, 9, 10, 5],
        ]

        pattern = FanPattern(center_point, radius_point, normal)
        ratios = [self.core_ratio, self.diagonal_ratio]
        angles = np.linspace(0, np.pi, num=5)

        super().__init__(
            [center_point, *pattern.get_inner_points(angles, ratios), *pattern.get_outer_points(angles)], quad_map
        )

    @property
    def grid(self):
        return [self.faces[:2], self.faces[2:]]


class FourCoreDisk(DiskBase):
    """A disk with four quads in the core and 8 in shell;
    the most versatile base for round objects."""

    chops: ClassVar = [[4], [4, 5, 6, 8]]

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        quad_map = [
            # core
            [0, 1, 2, 3],
            [5, 0, 3, 4],
            [6, 7, 0, 5],
            [7, 8, 1, 0],
            # shell
            [1, 9, 10, 2],
            [2, 10, 11, 3],
            [3, 11, 12, 4],
            [4, 12, 13, 5],
            [5, 13, 14, 6],
            [6, 14, 15, 7],
            [7, 15, 16, 8],
            [8, 16, 9, 1],
        ]

        pattern = FanPattern(center_point, radius_point, normal)
        ratios = [self.core_ratio, self.diagonal_ratio]
        angles = np.linspace(0, 2 * np.pi, num=8, endpoint=False)

        super().__init__(
            [center_point, *pattern.get_inner_points(angles, ratios), *pattern.get_outer_points(angles)], quad_map
        )

    @property
    def grid(self):
        return [self.faces[:4], self.faces[4:]]


Disk = FourCoreDisk


class WrappedDisk(DiskBase):
    """A OneCoreDisk but with four additional blocks surrounding it,
    making the sketch a square"""

    chops: ClassVar = [
        [6],
        [1, 2],
    ]

    def __init__(self, center_point: PointType, corner_point: PointType, radius: float, normal: VectorType):
        # TODO: make pattern a property, ready to be adjusted by subclasses
        pattern = FanPattern(center_point, corner_point, normal)
        angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)

        radius_ratio = radius / f.norm(pattern.radius_vector)
        square_ratio = self.diagonal_ratio * radius_ratio

        square_points = pattern.get_inner_points(angles, [square_ratio])
        arc_points = pattern.get_inner_points(angles, [radius_ratio])
        outer_points = pattern.get_outer_points(angles)

        quad_map = [
            # core
            [0, 1, 2, 3],
            # shell
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
            # with added outer quads
            [4, 8, 9, 5],
            [5, 9, 10, 6],
            [6, 10, 11, 7],
            [7, 11, 8, 4],
        ]

        super().__init__([*square_points, *arc_points, *outer_points], quad_map)
        self.origo_point = Point(center_point)

    @property
    def grid(self):
        return [[self.faces[0]], self.faces[1:5], self.faces[5:]]

    def add_edges(self):
        for face in self.grid[1]:
            face.add_edge(1, Origin(self.origo))

    @property
    def center(self):
        return self.faces[0].center


class Oval(DiskBase):
    chops: ClassVar = [
        [6],
        [6, 7, 8, 10, 11],
    ]

    def __init__(self, center_point_1: PointType, center_point_2: PointType, normal: VectorType, radius: float):
        quad_map = [
            # the core
            [0, 1, 2, 3],  # 0
            [5, 0, 3, 4],  # 1
            [7, 6, 0, 5],  # 2
            [8, 9, 6, 7],  # 3
            [9, 10, 11, 6],  # 4
            [6, 11, 1, 0],  # 5
            # the shell
            [1, 12, 13, 2],  # 6
            [2, 13, 14, 3],  # 7
            [3, 14, 15, 4],  # 8
            [4, 15, 16, 5],  # 9
            [5, 16, 17, 7],  # 10
            [7, 17, 18, 8],  # 11
            [8, 18, 19, 9],  # 12
            [9, 19, 20, 10],  # 13
            [10, 20, 21, 11],  # 14
            [11, 21, 12, 1],  # 15
        ]

        center_point_1 = np.array(center_point_1)
        center_point_2 = np.array(center_point_2)
        normal = f.unit_vector(np.asarray(normal))

        ratios = [self.core_ratio, self.diagonal_ratio]
        angles = np.linspace(0, np.pi, num=5)

        center_delta = center_point_2 - center_point_1
        radius_vector_1 = f.unit_vector(np.cross(normal, center_delta)) * radius
        radius_point_1 = center_point_1 + radius_vector_1  # point 12 on the sketch

        pattern_1 = FanPattern(center_point_1, radius_point_1, normal)

        radius_vector_2 = f.unit_vector(np.cross(normal, -center_delta)) * radius
        radius_point_2 = center_point_2 + radius_vector_2  # point 17 on the sketch
        pattern_2 = FanPattern(center_point_2, radius_point_2, normal)

        inner_points_1 = pattern_1.get_inner_points(angles, ratios)
        inner_points_2 = pattern_2.get_inner_points(angles, ratios)

        outer_points_1 = pattern_1.get_outer_points(angles)
        outer_points_2 = pattern_2.get_outer_points(angles)

        locations = [center_point_1, *inner_points_1, center_point_2, *inner_points_2, *outer_points_1, *outer_points_2]

        super().__init__(locations, quad_map)

    @property
    def center_1(self) -> NPPointType:
        return self.faces[0].points[0].position

    @property
    def center_2(self) -> NPPointType:
        return self.faces[5].points[0].position

    @property
    def center(self):
        return (self.center_1 + self.center_2) / 2

    def add_edges(self):
        for i in (6, 7, 8, 9):
            self.faces[i].add_edge(1, Origin(self.center_1))

        for i in (11, 12, 13, 14):
            self.faces[i].add_edge(1, Origin(self.center_2))

    @property
    def grid(self):
        return [self.faces[:6], self.faces[6:]]

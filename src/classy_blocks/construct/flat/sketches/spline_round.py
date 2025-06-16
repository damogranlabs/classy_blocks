from typing import ClassVar, Optional

import numpy as np

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType, PointType
from classy_blocks.construct.edges import Origin, Spline
from classy_blocks.construct.flat.sketches.disk import DiskBase, FourCoreDisk, HalfDisk, QuarterDisk
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class SplineRound(DiskBase):
    """
    Base class for spline round sketches.
    Shape can be oval, elliptical or circular.
    """

    n_outer_spline_points = 20
    n_straight_spline_points = 10

    # Widths only used for rings
    width_1: float = 0
    width_2: float = 0

    disk_initialized: bool = False

    def __init__(self, side_1: float, side_2: float, **kwargs):
        self.side_1 = side_1
        self.side_2 = side_2

        self.n_outer_spline_points = kwargs.get("n_outer_spline_points", self.n_outer_spline_points)
        self.n_straight_spline_points = kwargs.get("n_straight_spline_points", self.n_straight_spline_points)

    def remove_core(self):
        """Remove core. Used for rings"""
        # remove center point
        pos = self.positions
        pos = np.delete(pos, 0, axis=0)

        # Remove core face
        self._faces = np.delete(self._faces, slice(0, int(len(self.shell) / 2)), axis=0)
        self.indexes = np.delete(self.indexes, slice(0, int(len(self.shell) / 2)), axis=0) - 1

        self.update(pos)

    def oval_core_spline(
        self,
        p_core_ratio: PointType,
        p_diagonal_ratio: PointType,
        radius_1: float,
        side_1: float,
        radius_2: float,
        side_2: float,
        reverse: bool = False,
    ) -> NPPointListType:
        """Creates the spline points for the core."""
        p_0 = np.asarray(p_core_ratio)
        p_1 = np.asarray(p_diagonal_ratio)

        # Create unitary points of p_0 and p_1
        r_1 = radius_1 - side_1
        r_2 = radius_2 - side_2
        p_0_u = np.array([0, side_1 + self.core_ratio * r_1, 0])
        p_1_u = np.array(
            [0, side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1, side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2]
        )

        # In case of oval shape the center and p_0 used to get the curvy spline are adjusted
        center_u_adj = np.array([0, side_1, side_2])
        p_0_u_adj = p_0_u + np.array([0, 0, side_2])
        spline_points_u = self.circular_core_spline(p_0_u_adj, p_1_u, center=center_u_adj, reverse=reverse)

        # Add straight part for ovals
        if side_2 > constants.TOL:
            if reverse:
                side_points_u = np.linspace(
                    p_0_u_adj, p_0_u_adj - np.array([0, 0, 0.05 * side_2]), self.n_straight_spline_points
                )
                spline_points_u = np.append(spline_points_u, side_points_u, axis=0)
            else:
                side_points_u = np.linspace(
                    p_0_u_adj - np.array([0, 0, 0.05 * side_2]), p_0_u_adj, self.n_straight_spline_points
                )
                spline_points_u = np.insert(spline_points_u, 0, side_points_u, axis=0)

        # Orthogonal vectors based on p_0_u and p_1_u
        u_0_org = p_0_u
        u_1_org = p_1_u - np.dot(p_1_u, f.unit_vector(u_0_org)) * f.unit_vector(u_0_org)

        # Spline points in u_0_org and u_1_org
        spline_d_0_org = np.dot(spline_points_u, f.unit_vector(u_0_org)).reshape((-1, 1)) / f.norm(u_0_org)
        spline_d_1_org = np.dot(spline_points_u, f.unit_vector(u_1_org)).reshape((-1, 1)) / f.norm(u_1_org)

        # New plane defined by new points
        u_0 = p_0 - self.center
        u_1 = p_1 - self.center - np.dot(p_1 - self.center, f.unit_vector(u_0)) * f.unit_vector(u_0)

        spline_points_new = self.center + spline_d_0_org * u_0 + spline_d_1_org * u_1
        return spline_points_new

    def add_core_spline_edges(self) -> None:
        """Add a spline to the core blocks for an optimized mesh."""
        sides = [self.side_1, self.side_2]
        radi = [self.radius_1, self.radius_2]
        for i, face in enumerate(self.core):
            p_0 = face.point_array[(i + 1) % 4]  # Core point on radius 1
            p_1 = face.point_array[(i + 2) % 4]  # Core point on diagonal
            p_2 = face.point_array[(i + 3) % 4]  # Core point on radius 2

            curve_0_1 = self.oval_core_spline(
                p_0, p_1, radi[i % 2], sides[i % 2], radi[(i + 1) % 2], sides[(i + 1) % 2], reverse=i == 2
            )
            curve_1_2 = self.oval_core_spline(
                p_2, p_1, radi[(i + 1) % 2], sides[(i + 1) % 2], radi[i % 2], sides[i % 2], reverse=i != 1
            )

            spline_curve_0_1 = Spline(curve_0_1)
            spline_curve_1_2 = Spline(curve_1_2)

            # Add curves to edges
            edge_1 = (i + 1) % 4
            edge_2 = (i + 2) % 4
            face.add_edge(edge_1, spline_curve_0_1)
            face.add_edge(edge_2, spline_curve_1_2)

    def outer_spline(
        self,
        p_radius: PointType,
        p_diagonal: PointType,
        radius_1: float,
        side_1: float,
        radius_2: float,
        side_2: float,
        center: Optional[NPPointType] = None,
        reverse: bool = False,
    ) -> NPPointListType:
        """Creates the spline points for the core."""
        p_0 = np.asarray(p_radius)
        p_1 = np.asarray(p_diagonal)
        center = self.origo if center is None else np.asarray(center)

        # Create unitary points of p_0 and p_1
        r_1 = radius_1 - side_1
        r_2 = radius_2 - side_2
        p_0_u = np.array([0, radius_1, 0])
        p_1_u = np.array([0, side_1 + 2 ** (-1 / 2) * r_1, side_2 + 2 ** (-1 / 2) * r_2])

        p_0_u_adj = p_0_u + np.array([0, 0, side_2])
        c_0_u_adj = np.array([0, side_1, side_2])

        theta = np.linspace(0, np.pi / 4, self.n_outer_spline_points)
        spline_points_u = c_0_u_adj + np.array([np.zeros(len(theta)), r_1 * np.cos(theta), r_2 * np.sin(theta)]).T

        if reverse:
            spline_points_u = spline_points_u[::-1]
            # Add straight part for ovals
            if side_2 > constants.TOL:
                side_points_u = np.linspace(
                    p_0_u_adj, p_0_u_adj - np.array([0, 0, 0.05 * side_2]), self.n_straight_spline_points
                )
                spline_points_u = np.append(spline_points_u, side_points_u, axis=0)
        else:
            # Add straight part for ovals
            if side_2 > constants.TOL:
                side_points_u = np.linspace(
                    p_0_u_adj - np.array([0, 0, 0.05 * side_2]), p_0_u_adj, self.n_straight_spline_points
                )
                spline_points_u = np.insert(spline_points_u, 0, side_points_u, axis=0)

        # Orthogonal vectors based on p_0_u and p_1_u
        u_0_org = p_0_u
        u_1_org = p_1_u - np.dot(p_1_u, f.unit_vector(u_0_org)) * f.unit_vector(u_0_org)

        # Spline points in u_0_org and u_1_org
        spline_d_0_org = np.dot(spline_points_u, f.unit_vector(u_0_org)).reshape((-1, 1)) / f.norm(u_0_org)
        spline_d_1_org = np.dot(spline_points_u, f.unit_vector(u_1_org)).reshape((-1, 1)) / f.norm(u_1_org)

        # New plane defined by new points
        u_0 = p_0 - center
        u_1 = p_1 - center - np.dot(p_1 - center, f.unit_vector(u_0)) * f.unit_vector(u_0)

        spline_points_new = center + spline_d_0_org * u_0 + spline_d_1_org * u_1
        return spline_points_new

    def add_outer_spline_edges(self, center: Optional[NPPointType] = None) -> None:
        """Add curved edge as spline to outside of sketch"""
        sides = [self.side_1, self.side_2]
        radi = [self.radius_1, self.radius_2]
        for i, face in enumerate(self.shell):
            p_0 = face.point_array[(i % 2) + 1]  # Outer point on radius
            p_1 = face.point_array[((i + 1) % 2) + 1]  # Outer point on diagonal

            spline_curve_0_1 = self.outer_spline(
                p_0,
                p_1,
                radi[int((i + 1) / 2) % 2],
                sides[int((i + 1) / 2) % 2],
                radi[int((i + 3) / 2) % 2],
                sides[int((i + 3) / 2) % 2],
                center,
                reverse=i % 2 == 1,
            )
            face.add_edge(1, Spline(spline_curve_0_1))

    def add_inner_spline_edges(self, center: Optional[NPPointType] = None) -> None:
        """Add curved edge as spline to inside of ring"""
        sides = [self.side_1, self.side_2, self.side_2, self.side_1]
        radi = [
            self.radius_1 - self.width_1,
            self.radius_2 - self.width_2,
            self.radius_2 - self.width_2,
            self.radius_1 - self.width_1,
        ]
        for i, face in enumerate(self.shell):
            p_0 = face.point_array[0 if i % 2 == 0 else 3]  # Inner point on radius
            p_1 = face.point_array[3 if i % 2 == 0 else 0]  # Inner point on diagonal

            spline_curve_0_1 = self.outer_spline(
                p_0, p_1, radi[i % 4], sides[i % 4], radi[(i + 1) % 4], sides[(i + 1) % 4], center, reverse=i % 2 == 1
            )

            face.add_edge(3, Spline(spline_curve_0_1))

    def add_edges(self) -> None:
        # Don't run add_edges in QuarterDisk.__init__()
        if not self.disk_initialized:
            return
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
        ):
            super().add_edges()
        else:
            self.add_core_spline_edges()
            self.add_outer_spline_edges()

    @property
    def radius_1_point(self) -> NPPointType:
        return self.radius_point

    @property
    def radius_1_vector(self) -> NPVectorType:
        return self.radius_1_point - self.origo

    @property
    def radius_1(self) -> float:
        return self.radius

    @property
    def radius_2_point(self) -> NPPointType:
        return self.shell[1].points[2].position

    @property
    def radius_2_vector(self) -> NPVectorType:
        return self.radius_2_point - self.origo

    @property
    def radius_2(self) -> float:
        return float(f.norm(self.radius_2_vector))

    @property
    def u_0(self) -> NPVectorType:
        """Returns unit vector 0 in stable way after transforms."""
        return f.unit_vector(np.cross(self.u_1, self.u_2))

    @property
    def normal(self) -> NPVectorType:
        return self.u_0

    @property
    def u_1(self) -> NPVectorType:
        """Returns unit vector 1 in stable way after transforms."""
        try:
            return f.unit_vector(self.radius_1_vector)
        except AttributeError:
            return self._u_1

    @u_1.setter
    def u_1(self, vec):
        self._u_1 = f.unit_vector(vec)

    @property
    def u_2(self) -> NPVectorType:
        """Returns unit vector 2 in stable way after transforms."""
        try:
            return f.unit_vector(self.radius_2_vector)
        except AttributeError:
            return self._u_2

    @u_2.setter
    def u_2(self, vec):
        self._u_2 = f.unit_vector(vec)

    def scale(self, ratio: float, origin: Optional[PointType] = None):
        """Reimplementation of scale to include side_1 and side_2."""
        self.side_1 = ratio * self.side_1
        self.side_2 = ratio * self.side_2

        return super().scale(ratio, origin)


class QuarterSplineDisk(SplineRound, QuarterDisk):
    """Sketch for Quarter oval, elliptical and circular shapes"""

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs,
    ) -> None:
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        note the vectors from the center to corner 1 and 2 should be perpendicular.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
        """
        super().__init__(side_1, side_2, **kwargs)

        corner_1_point = np.asarray(corner_1_point)
        corner_2_point = np.asarray(corner_2_point)
        self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
        self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))

        # Create a QuarterDisk
        self.disk_initialized = False
        super(SplineRound, self).__init__(center_point, corner_1_point, normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)
        self.add_edges()

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        # Core
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[2] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2

        # Shell
        pos[5] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )
        pos[6] = corner_2_point

        self.update(pos)


class HalfSplineDisk(SplineRound, HalfDisk):
    """Sketch for Half oval, elliptical and circular shapes"""

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs,
    ) -> None:
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        note the vectors from the center to corner 1 and 2 should be perpendicular.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
        """
        super().__init__(side_1, side_2, **kwargs)

        corner_1_point = np.asarray(corner_1_point)
        corner_2_point = np.asarray(corner_2_point)
        self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
        self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))

        # Create a HalfDisk
        self.disk_initialized = False
        super(SplineRound, self).__init__(center_point, corner_1_point, normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)
        self.add_edges()

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        # Core
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[2] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[4] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[5] = self.center - (self.side_1 + self.core_ratio * r_1) * self.u_1

        # Shell
        pos[7] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )
        pos[8] = corner_2_point
        pos[9] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )

        self.update(pos)


class SplineDisk(SplineRound, FourCoreDisk):
    """Sketch for oval, elliptical and circular shapes"""

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs,
    ) -> None:
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        note the vectors from the center to corner 1 and 2 should be perpendicular.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
        """
        super().__init__(side_1, side_2, **kwargs)

        corner_1_point = np.asarray(corner_1_point)
        corner_2_point = np.asarray(corner_2_point)
        self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
        self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))

        # Create a FourCoreDisk
        self.disk_initialized = False
        super(SplineRound, self).__init__(center_point, corner_1_point, normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)
        self.add_edges()

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        # Core
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[2] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[4] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[5] = self.center - (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[6] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            - (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )
        pos[7] = self.center - (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[8] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1
            - (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        )

        # Shell
        pos[9] = corner_1_point
        pos[10] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )
        pos[11] = corner_2_point
        pos[12] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            + (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )
        pos[13] = self.center - (self.side_1 + r_1) * self.u_1
        pos[14] = (
            self.center
            - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            - (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )
        pos[15] = self.center - (self.side_2 + r_2) * self.u_2
        pos[16] = (
            self.center
            + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1
            - (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        )

        self.update(pos)


class QuarterSplineRing(QuarterSplineDisk):
    """Ring based on SplineRound."""

    chops: ClassVar = [
        [0],  # axis 0
        [0, 1],  # axis 1
    ]

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        width_1: float,
        width_2: float,
        **kwargs,
    ):
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        Note the vectors from the center to corner 1 and 2 should be perpendicular.
        The ring is defined such it will fit around a QuaterSplineRound defined with the same center, corners and sides.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
            width_1: Width of shell
            width_2: Width of shell
        """
        self.width_1 = float(width_1)
        self.width_2 = float(width_2)

        # Convert to corners to outer corners
        if kwargs.get("from_inner", True):
            corner_1_point = np.asarray(corner_1_point)
            corner_2_point = np.asarray(corner_2_point)
            self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
            self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))
            corner_1_point = corner_1_point + self.width_1 * self.u_1
            corner_2_point = corner_2_point + self.width_2 * self.u_2

        # Initialize QuarterDisk
        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2, **kwargs)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a disk to a ting"""

        # First adjust circular disk to SplineDisk
        super().correct_disk(corner_1_point, corner_2_point)
        self.remove_core()

        # Adjust inner curve to be oval/elliptical
        r_1 = self.radius_1 - self.side_1 - self.width_1
        r_2 = self.radius_2 - self.side_2 - self.width_2

        pos = self.positions
        pos[0] = self.radius_1_point - self.width_1 * self.u_1
        pos[1] = (
            self.origo
            + (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            + (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[2] = self.radius_2_point - self.width_2 * self.u_2
        self.update(pos)

    def add_edges(self) -> None:
        # Don't run add_edges in QuarterDisk.__init__()
        if not self.disk_initialized:
            return

        # Outside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(1, Origin(self.origo))
        else:
            self.add_outer_spline_edges()

        # Inside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
            and abs(self.width_1 - self.width_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(3, Origin(self.origo))
        else:
            self.add_inner_spline_edges()

    @property
    def grid(self):
        return [self.faces[-2:]]

    @property
    def core(self):
        return None


class HalfSplineRing(HalfSplineDisk):
    """Ring based on SplineRound."""

    chops: ClassVar = [
        [0],  # axis 0
        [0, 1, 2, 3],  # axis 1
    ]

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        width_1: float,
        width_2: float,
        **kwargs,
    ):
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        Note the vectors from the center to corner 1 and 2 should be perpendicular.
        The ring is defined such it will fit around a QuaterSplineRound defined with the same center, corners and sides.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
            width_1: Width of shell
            width_2: Width of shell
        """
        self.width_1 = float(width_1)
        self.width_2 = float(width_2)

        # Convert to corners to outer corners
        if kwargs.get("from_inner", True):
            corner_1_point = np.asarray(corner_1_point)
            corner_2_point = np.asarray(corner_2_point)
            self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
            self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))
            corner_1_point = corner_1_point + self.width_1 * self.u_1
            corner_2_point = corner_2_point + self.width_2 * self.u_2

        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2, **kwargs)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a disk to a ting"""

        # First adjust circular disk to SplineDisk
        super().correct_disk(corner_1_point, corner_2_point)
        self.remove_core()

        # Adjust inner curve to be oval/elliptical
        r_1 = self.radius_1 - self.side_1 - self.width_1
        r_2 = self.radius_2 - self.side_2 - self.width_2

        pos = self.positions
        pos[0] = self.radius_1_point - self.width_1 * self.u_1
        pos[1] = (
            self.origo
            + (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            + (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[2] = self.radius_2_point - self.width_2 * self.u_2
        pos[3] = (
            self.origo
            - (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            + (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[4] = self.origo - (self.radius_1 - self.width_1) * self.u_1
        self.update(pos)

    def add_edges(self) -> None:
        # Don't run add_edges in QuarterDisk.__init__()
        if not self.disk_initialized:
            return

        # Outside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(1, Origin(self.origo))
        else:
            self.add_outer_spline_edges()

        # Inside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
            and abs(self.width_1 - self.width_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(3, Origin(self.origo))
        else:
            self.add_inner_spline_edges()

    @property
    def grid(self):
        return [self.faces[-4:]]

    @property
    def core(self):
        return None


class SplineRing(SplineDisk):
    """Ring based on SplineRound."""

    chops: ClassVar = [
        [0],  # axis 0
        [0, 1, 2, 3, 4, 5, 6, 7],  # axis 1
    ]

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        width_1: float,
        width_2: float,
        **kwargs,
    ):
        """
        With a normal in x direction corner 1 will be in the y direction and corner 2 the z direction.
        Note the vectors from the center to corner 1 and 2 should be perpendicular.
        The ring is defined such it will fit around a QuaterSplineRound defined with the same center, corners and sides.
        Args:
            center_point: Center of round shape
            corner_1_point: Radius for circular and elliptical shape
            corner_2_point: Radius for circular and elliptical  shape
            side_1: Straight length for oval shape
            side_2: Straight length for oval shape
            width_1: Width of shell
            width_2: Width of shell
        """
        self.width_1 = float(width_1)
        self.width_2 = float(width_2)

        # Convert to corners to outer corners
        if kwargs.get("from_inner", True):
            corner_1_point = np.asarray(corner_1_point)
            corner_2_point = np.asarray(corner_2_point)
            self.u_1 = f.unit_vector(corner_1_point - np.asarray(center_point))
            self.u_2 = f.unit_vector(corner_2_point - np.asarray(center_point))
            corner_1_point = corner_1_point + self.width_1 * self.u_1
            corner_2_point = corner_2_point + self.width_2 * self.u_2

        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2, **kwargs)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a disk to a ting"""

        # First adjust circular disk to splinedisk
        super().correct_disk(corner_1_point, corner_2_point)
        self.remove_core()

        # Adjust inner curve to be oval/elliptical
        r_1 = self.radius_1 - self.side_1 - self.width_1
        r_2 = self.radius_2 - self.side_2 - self.width_2

        pos = self.positions
        pos[0] = self.radius_1_point - self.width_1 * self.u_1
        pos[1] = (
            self.origo
            + (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            + (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[2] = self.radius_2_point - self.width_2 * self.u_2
        pos[3] = (
            self.origo
            - (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            + (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[4] = self.origo - (self.radius_1 - self.width_1) * self.u_1
        pos[5] = (
            self.origo
            - (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            - (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        pos[6] = self.origo - (self.radius_2 - self.width_2) * self.u_2
        pos[7] = (
            self.origo
            + (self.side_1 + r_1 * np.cos(np.pi / 4)) * self.u_1
            - (self.side_2 + r_2 * np.sin(np.pi / 4)) * self.u_2
        )
        self.update(pos)

    def add_edges(self) -> None:
        # Don't run add_edges in QuarterDisk.__init__()
        if not self.disk_initialized:
            return

        # Outside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(1, Origin(self.origo))
        else:
            self.add_outer_spline_edges()

        # Inside
        # Circular
        if (
            self.side_1 < constants.TOL
            and self.side_2 < constants.TOL
            and abs(self.radius_1 - self.radius_2) < constants.TOL
            and abs(self.width_1 - self.width_2) < constants.TOL
        ):
            for face in self.shell:
                face.add_edge(3, Origin(self.origo))
        else:
            self.add_inner_spline_edges()

    @property
    def grid(self):
        return [self.faces[-8:]]

    @property
    def core(self):
        return None

from typing import ClassVar, List, Optional
import inspect
import numpy as np

from classy_blocks.construct.edges import Origin, Spline
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.disk import DiskBase, QuarterDisk, HalfDisk, FourCoreDisk
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.point import Point
from classy_blocks.types import NPPointType, NPVectorType, PointType, NPPointListType
from classy_blocks.util import constants
from classy_blocks.util.constants import TOL
from classy_blocks.util import functions as f
from classy_blocks.base import transforms as tr


class SplineRound(DiskBase):
    """
    Base class for spline round sketches.
    Shape can be oval, elliptical or circular.
    """

    n_outer_spline_points = 20
    n_straight_spline_points = 10

    def __init__(
        self,
        side_1: float,
        side_2: float,
        **kwargs
    ):
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
        self.side_1 = side_1
        self.side_2 = side_2

        self.n_outer_spline_points = kwargs.get('n_outer_spline_points', self.n_outer_spline_points)
        self.n_straight_spline_points = kwargs.get('n_straight_spline_points', self.n_straight_spline_points)

    def core_spline(self, p_core_ratio: PointType, p_diagonal_ratio: PointType, radius_1, side_1, radius_2, side_2,
                    reverse: bool = False) -> NPPointListType:
        """Creates the spline points for the core."""
        p_0 = np.asarray(p_core_ratio)
        p_1 = np.asarray(p_diagonal_ratio)

        r_1 = radius_1 - side_1
        r_2 = radius_2 - side_2
        p_0_u = np.array([0, side_1 + self.core_ratio * r_1, 0])
        p_1_u = np.array([0, side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1,
                          side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2])

        center_u_adj = np.array([0, side_1, side_2])
        p_0_u_adj = p_0_u + np.array([0, 0, side_2])

        spline_points_u = super().core_spline(p_0_u_adj, p_1_u, center=center_u_adj, reverse=reverse)

        if side_2 > constants.TOL:
            if reverse:
                side_points_u = np.linspace(p_0_u_adj, p_0_u_adj - np.array([0, 0, 0.05 * side_2]), self.n_straight_spline_points)
                spline_points_u = np.append(spline_points_u, side_points_u, axis=0)
            else:
                side_points_u = np.linspace(p_0_u_adj - np.array([0, 0, 0.05 * side_2]), p_0_u_adj, self.n_straight_spline_points)
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
        shifts = [self.side_2 * self.u_2, -self.side_1 * self.u_1, -self.side_2 * self.u_2, self.side_1 * self.u_1]
        for i, face in enumerate(self.core):
            p_0 = face.point_array[(i + 1) % 4]     # Core point on radius 1
            p_1 = face.point_array[(i + 2) % 4]     # Core point on diagonal
            p_2 = face.point_array[(i + 3) % 4]     # Core point on radius 2
            #print(i, p_0, p_1, p_2)

            p_0 += shifts[i]
            p_2 += shifts[i-1]
            center = self.center + shifts[i-1] + shifts[i]

            """
            if i == 0:
                center = self.center + self.side_1 * self.u_1 + self.side_2 * self.u_2
                p_0 += self.side_2 * self.u_2
                p_2 += self.side_1 * self.u_1
            elif i == 1:
                center = self.center - self.side_1 * self.u_1 + self.side_2 * self.u_2
                p_0 += - self.side_1 * self.u_1
                p_2 += self.side_2 * self.u_2
            elif i == 2:
                center = self.center - self.side_1 * self.u_1 - self.side_2 * self.u_2
                p_0 += - self.side_2 * self.u_2
                p_2 += - self.side_1 * self.u_1
            elif i == 3:
                center = self.center + self.side_1 * self.u_1 - self.side_2 * self.u_2
                p_0 += + self.side_1 * self.u_1
                p_2 += - self.side_2 * self.u_2
            """

            # print(i, p_0, p_1, p_2)
            curve_0_1 = super().core_spline(p_0, p_1, reverse=i==2, center=center)
            curve_1_2 = super().core_spline(p_2, p_1, reverse=i!=1, center=center)
            if self.side_2 > constants.TOL:
                if i == 0:
                    core_side_points = np.linspace(p_0 - 0.05 * self.side_2 * self.u_2, p_0, self.n_straight_spline_points)
                    curve_0_1 = np.insert(curve_0_1, 0, core_side_points, axis=0)
                elif i == 1:
                    core_side_points = np.linspace(p_2 - 0.05 * self.side_2 * self.u_2, p_2, self.n_straight_spline_points)
                    curve_1_2 = np.insert(curve_1_2, 0, core_side_points, axis=0)
                elif i == 2:
                    core_side_points = np.linspace(p_0, p_0 + 0.05 * self.side_2 * self.u_2, self.n_straight_spline_points)
                    curve_0_1 = np.append(curve_0_1, core_side_points, axis=0)
                elif i == 3:
                    core_side_points = np.linspace(p_2, p_2 + 0.05 * self.side_2 * self.u_2, self.n_straight_spline_points)
                    curve_1_2 = np.append(curve_1_2, core_side_points, axis=0)


            if self.side_1 > constants.TOL:
                if i == 0:
                    core_side_points = np.linspace(p_2, p_2 - 0.05 * self.side_1 * self.u_1, self.n_straight_spline_points)
                    curve_1_2 = np.append(curve_1_2, core_side_points, axis=0)
                elif i == 1:
                    core_side_points = np.linspace(p_0 + 0.05 * self.side_1 * self.u_1, p_0, self.n_straight_spline_points)
                    curve_0_1 = np.insert(curve_0_1, 0, core_side_points, axis=0)
                elif i == 2:
                    core_side_points = np.linspace(p_2, p_2 + 0.05 * self.side_1 * self.u_1, self.n_straight_spline_points)
                    curve_1_2 = np.append(curve_1_2, core_side_points, axis=0)
                elif i == 3:
                    core_side_points = np.linspace(p_0 - 0.05 * self.side_1 * self.u_1, p_0, self.n_straight_spline_points)
                    curve_0_1 = np.insert(curve_0_1, 0, core_side_points, axis=0)
            sides = [self.side_1, self.side_2]
            radi = [self.radius_1, self.radius_2]
            print(i)
            print(abs(curve_0_1 - self.core_spline(face.point_array[(i + 1) % 4],
                                                   face.point_array[(i + 2) % 4],
                                                   radi[i%2], sides[i%2],
                                                   radi[(i+1)%2], sides[(i+1)%2],
                                                   reverse=i==2)).all()<1e-15)

            print(abs(curve_1_2 - self.core_spline(face.point_array[(i + 3) % 4],
                                                   face.point_array[(i + 2) % 4],
                                                   radi[(i+1)%2], sides[(i+1)%2],
                                                   radi[i%2], sides[i%2],
                                                   reverse=i!=1)).all()<1e-15)
            """
            if i == 0:
                print(i)
                print(abs(curve_0_1 - self.core_spline(face.point_array[(i + 1) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       radi[i%2], sides[i%2],
                                                       radi[(i+1)%2], sides[(i+1)%2],
                                                       reverse=i==2)).all()<1e-15)

                print(abs(curve_1_2 - self.core_spline(face.point_array[(i + 3) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       radi[(i+1)%2], sides[(i+1)%2],
                                                       radi[i%2], sides[i%2],
                                                       reverse=i!=1)).all()<1e-15)

            elif i == 1:
                print(i)
                print(abs(curve_0_1 - self.core_spline(face.point_array[(i + 1) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_2, self.side_2,
                                                       self.radius_1,self.side_1,
                                                       reverse=i==2)).all()<1e-15)

                print(abs(curve_1_2 - self.core_spline(face.point_array[(i + 3) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_1, self.side_1,
                                                       self.radius_2, self.side_2,
                                                       reverse=i!=1)).all()<1e-15)

            if i == 2:
                print(i)
                print(abs(curve_0_1 - self.core_spline(face.point_array[(i + 1) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_1, self.side_1,
                                                       self.radius_2,self.side_2,
                                                       reverse=i==2)).all()<1e-15)

                print(abs(curve_1_2 - self.core_spline(face.point_array[(i + 3) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_2, self.side_2,
                                                       self.radius_1, self.side_1,
                                                       reverse=i!=1)).all()<1e-15)
            elif i == 3:
                print(i)
                print(abs(curve_0_1 - self.core_spline(face.point_array[(i + 1) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_2, self.side_2,
                                                       self.radius_1,self.side_1,
                                                       reverse=i==2)).all()<1e-15)

                print(abs(curve_1_2 - self.core_spline(face.point_array[(i + 3) % 4],
                                                       face.point_array[(i + 2) % 4],
                                                       self.radius_1, self.side_1,
                                                       self.radius_2, self.side_2,
                                                       reverse=i!=1)).all()<1e-15)

            """
            #print(i, p_0, curve_0_1, p_1)
            #print(i, p_1, curve_1_2, p_2)
            curve_0_1 = Spline(curve_0_1)
            curve_1_2 = Spline(curve_1_2)
            # Add curves to edges
            edge_1 = (i + 1) % 4
            edge_2 = (i + 2) % 4
            face.add_edge(edge_1, curve_0_1)
            face.add_edge(edge_2, curve_1_2)

    def add_outer_spline_edges(self) -> None:
        for i, face in enumerate(self.shell):
            theta = np.linspace(i * np.pi / 4, (i + 1)*np.pi / 4,
                                self.n_outer_spline_points + 1, endpoint=False)[1:].reshape((-1, 1))
            shell_curve_points = (self.center +
                                  (self.side_1 + self.r_1 * np.cos(theta)) * self.u_1 +
                                  (self.side_2 + self.r_2 * np.sin(theta)) * self.u_2)
            if i % 2 == 0 and self.side_2 > constants.TOL:
                outer_side_points = np.linspace(
                    self.radius_1_point + 0.95 * self.side_2 * self.u_2,
                    self.radius_1_point + self.side_2 * self.u_2,
                    self.n_straight_spline_points
                )
                shell_curve_points = np.insert(shell_curve_points, 0, outer_side_points, axis=0)
            elif i % 2 == 1 and self.side_1 > constants.TOL:
                outer_side_points = np.linspace(
                    self.radius_2_point + self.side_1 * self.u_1,
                    self.radius_2_point + 0.95 * self.side_1 * self.u_1,
                    self.n_straight_spline_points)
                shell_curve_points = np.append(shell_curve_points, outer_side_points, axis=0)
            face.add_edge(1, Spline(shell_curve_points))

    def add_edges(self) -> None:
        # Don't run add_edges in QuarterDisk.__init__()
        if not self.disk_initialized:
            return
        # Circular
        if self.side_1 < constants.TOL and self.side_2 < constants.TOL and abs(self.radius_1 - self.radius_2) < constants.TOL:
            super().add_edges()
        else:
            self.add_core_spline_edges()
            #self.add_outer_spline_edges()


    @property
    def radius_1_point(self) -> NPPointType:
        return self.radius_point

    @property
    def radius_1_vector(self) -> NPVectorType:
        return self.radius_vector

    @property
    def radius_1(self) -> float:
        return self.radius

    @property
    def radius_2_point(self) -> NPPointType:
        return self.grid[1][1].points[2].position

    @property
    def radius_2_vector(self) -> NPVectorType:
        return self.radius_2_point - self.center

    @property
    def radius_2(self) -> float:
        return float(f.norm(self.radius_2_vector))

    @property
    def r_1(self) -> float:
        return self.radius_1 - self.side_1

    @property
    def r_2(self) -> float:
        return self.radius_2 - self.side_2

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


    # @classmethod
    # def init_from_radius(cls, center_point, corner_1_point, corner_2_point, r_1, r_2):
    #     """Calculate the side lengths based on the radius and return sketch"""
    #     side_1 = f.norm(corner_1_point - center_point) - r_1
    #     side_2 = f.norm(corner_2_point - center_point) - r_2

    #     return cls(center_point, corner_1_point, corner_2_point, side_1, side_2)


class QuarterSplineDisk(SplineRound, QuarterDisk):
    """Sketch for Quarter oval, elliptical and circular shapes"""

    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs
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

        # Create a QuarterDisk and update positions
        self.disk_initialized = False
        super(SplineRound, self). __init__(center_point,corner_1_point,normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        # Note that self.r_2 here would give wrong results as corner_2 have not been updated
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        pos[-1] = corner_2_point
        pos[-2] = self.center + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 + \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[2] = self.center + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 + \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1
        self.update(pos)
        self.add_edges()


class HalfSplineDisk(SplineRound, HalfDisk):
    """Sketch for Half oval, elliptical and circular shapes"""
    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs
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

        # Create a HalfDisk and update positions
        self.disk_initialized = False
        super(SplineRound, self). __init__(center_point, corner_1_point, normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        # Note that self.r_2 here would give wrong results as corner_2 have not been updated
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        pos[-2] = self.center - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 + \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        pos[-3] = corner_2_point
        pos[-4] = self.center + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 + \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2

        pos[5] = self.center - (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[4] = self.center - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 + \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[2] = self.center + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 + \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1

        self.update(pos)
        self.add_edges()


class SplineDisk(SplineRound, FourCoreDisk):
    """Sketch for oval, elliptical and circular shapes"""
    def __init__(
        self,
        center_point: PointType,
        corner_1_point: PointType,
        corner_2_point: PointType,
        side_1: float,
        side_2: float,
        **kwargs
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

        # Create a HalfDisk and update positions
        self.disk_initialized = False
        super(SplineRound, self). __init__(center_point, corner_1_point, normal=self.u_0)
        self.disk_initialized = True

        # Adjust to actual shape
        self.correct_disk(corner_1_point, corner_2_point)

    def correct_disk(self, corner_1_point: NPPointType, corner_2_point: NPPointType):
        """Method to convert a circular disk to the elliptical/oval shape defined"""
        # Note that self.r_2 here would give wrong results as corner_2 have not been updated
        r_1 = f.norm(corner_1_point - self.center) - self.side_1
        r_2 = f.norm(corner_2_point - self.center) - self.side_2

        pos = self.positions
        # Core
        pos[1] = self.center + (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[2] = self.center + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 + \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[3] = self.center + (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[4] = self.center - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 + \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[5] = self.center - (self.side_1 + self.core_ratio * r_1) * self.u_1
        pos[6] = self.center - (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 - \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2
        pos[7] = self.center - (self.side_2 + self.core_ratio * r_2) * self.u_2
        pos[8] = self.center + (self.side_1 + 2 ** (-1 / 2) * self.diagonal_ratio * r_1) * self.u_1 - \
                 (self.side_2 + 2 ** (-1 / 2) * self.diagonal_ratio * r_2) * self.u_2

        # Shell
        pos[9] = corner_1_point
        pos[10] = self.center + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 + \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        pos[11] = corner_2_point
        pos[12] = self.center - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 + \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        pos[13] = self.center - (self.side_1 + r_1) * self.u_1
        pos[14] = self.center - (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 - \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2
        pos[15] = self.center - (self.side_2 + r_2) * self.u_2
        pos[16] = self.center + (self.side_1 + 2 ** (-1 / 2) * r_1) * self.u_1 - \
                  (self.side_2 + 2 ** (-1 / 2) * r_2) * self.u_2

        self.update(pos)
        self.add_edges()


class QuarterSplineRing(SplineRound):
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
        **kwargs
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
        super().__init__(side_1, side_2, **kwargs)

        center = np.array(center_point)
        self._center = Point(center)
        corner_1 = np.array(corner_1_point)
        corner_2 = np.array(corner_2_point)

        self.width_1 = float(width_1)
        self.width_2 = float(width_2)
        self._center = Point(center_point)

        # TODO: DRY
        u_1 = f.unit_vector(corner_1 - center)
        u_2 = f.unit_vector(corner_2 - center)

        r_1 = f.norm(corner_1 - center) - self.side_1
        r_2 = f.norm(corner_2 - center) - self.side_2

        r_1_outer = f.norm(corner_1 - center) - self.side_1 + self.width_1
        r_2_outer = f.norm(corner_2 - center) - self.side_2 + self.width_2

        p2 = corner_2
        p2_2 = corner_2 + self.width_2 * u_2
        p5 = corner_1
        p5_2 = corner_1 + self.width_1 * u_1
        p6 = center + (self.side_1 + 2 ** (-1 / 2) * r_1) * u_1 + (self.side_2 + 2 ** (-1 / 2) * r_2) * u_2
        p6_2 = center + (self.side_1 + 2 ** (-1 / 2) * r_1_outer) * u_1 + (self.side_2 + 2 ** (-1 / 2) * r_2_outer) * u_2

        quad_map = [
            [2, 3, 1, 0],
            [4, 5, 3, 2],
        ]

        positions = [p2, p2_2, p6, p6_2, p5, p5_2]
        super(SplineRound, self).__init__(positions, quad_map)

    @property
    def center(self):
        return self._center.position

    @property
    def corner_1(self) -> NPPointType:
        return self.faces[1].points[0].position

    @property
    def corner_2(self) -> NPPointType:
        return self.faces[0].points[3].position

    @property
    def r_1_outer(self) -> float:
        """Returns radius 1 in stable way after transforms."""
        return f.norm(self.corner_1 - self.center) - self.side_1 + self.width_1

    @property
    def r_2_outer(self) -> float:
        """Returns radius 2 in stable way after transforms."""
        return f.norm(self.corner_2 - self.center) - self.side_2 + self.width_2

    @property
    def grid(self):
        return [self.faces]

    @property
    def core(self):
        return None

    @property
    def shell(self):
        return self.grid[0]

    @property
    def parts(self):
        return [*super().parts, self._center]

    def scale(self, ratio: float, origin: Optional[PointType] = None):
        """Reimplementation of scale to include side_1 and side_2."""

        self.side_1 = ratio * self.side_1
        self.side_2 = ratio * self.side_2

        self.width_1 = ratio * self.width_1
        self.width_2 = ratio * self.width_2

        return super().scale(ratio, origin)

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

class HalfSplineRing(QuarterSplineRing):
    """Sketch for Half oval, elliptical and circular ring"""
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
        **kwargs
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
            width_1: Width of shell
            width_2: Width of shell
        """

        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2, width_1, width_2, **kwargs)
        other_quarter = QuarterSplineRing(self.center, self.corner_2, 2 * self.center - self.corner_1,
                                          side_2, side_1, width_2, width_1, **kwargs)
        self.merge(other_quarter)


class SplineRing(HalfSplineRing):
    """Sketch for full oval, elliptical and circular shapes"""
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
        **kwargs
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

        super().__init__(center_point, corner_1_point, corner_2_point, side_1, side_2, width_1, width_2, **kwargs)
        other_half = self.copy().transform([tr.Rotation(self.normal, np.pi, self.center)])
        self.merge(other_half)


if __name__ == '__main__':
    from classy_blocks.construct.shape import LoftedShape
    from classy_blocks.mesh import Mesh

    sketch1 = SplineDisk([0,0.2,0], [0,1,0], [0,0,2], side_1=0, side_2=1)

    sketch2 = SplineDisk([5,0.2,0], [5,1,0], [5,0,2], side_1=0.5, side_2=1)

    shape = LoftedShape(sketch1, sketch2)
    # chop radial
    shape.chop(0, count=10)
    shape.chop(1, count=12)
    shape.chop(2, count=14)

    mesh = Mesh()
    mesh.add(shape)
    mesh.write('C:/Users/LAHN/OneDrive - Kamstrup A S/Documents/pythonScripts/classy_blocks/examples/case/system/blockMeshDict', debug_path='debug.vtk')



import abc

import numpy as np

from classy_blocks.cbtyping import FloatListType, PointType
from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.edges import Spline
from classy_blocks.construct.flat.sketches.disk import HalfDisk
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.shapes.cylinder import SemiCylinder
from classy_blocks.util import functions as f


class CuspSemiCylinder(SemiCylinder):
    """A cylinder with one slanted/inclined end face"""

    sketch_class = HalfDisk

    def __init__(
        self, axis_point_1: PointType, axis_point_2: PointType, radius_point_1: PointType, end_angle: float = np.pi / 4
    ):
        axis_point_1 = np.asarray(axis_point_1)
        axis_point_2 = np.asarray(axis_point_2)
        axis = axis_point_2 - axis_point_1
        radius_point_1 = np.asarray(radius_point_1)
        radius_vector = radius_point_1 - axis_point_1

        shear_normal = np.cross(-axis, radius_vector)

        super().__init__(axis_point_1, axis_point_2, radius_point_1)

        for loft in self.operations:
            loft.top_face.remove_edges()

        for loft in self.shell:
            point_1 = loft.top_face.points[1].position
            point_2 = loft.top_face.points[2].position
            edge_points = f.divide_arc(axis_point_2, point_1, point_2, 5)
            loft.top_face.add_edge(1, Spline(edge_points))

        for loft in self.operations:
            loft.top_face.shear(shear_normal, axis_point_2, -axis, end_angle)

        # TODO: include those inner edges (Disk.spline_ratios > Curve)
        # self.remove_inner_edges(end=True)


class CuspCylinder(Assembly):
    def __init__(
        self,
        axis_point_1: PointType,
        axis_point_2: PointType,
        radius_point: PointType,
        end_angle_left: float,
        end_angle_right: float,
    ):
        axis_point_1 = np.asarray(axis_point_1)
        radius_point_right = np.asarray(radius_point)
        radius_vector_right = np.asarray(radius_point) - axis_point_1
        radius_point_left = axis_point_1 - radius_vector_right

        self.cusp_right = CuspSemiCylinder(axis_point_1, axis_point_2, radius_point_right, end_angle_right)
        self.cusp_left = CuspSemiCylinder(axis_point_1, axis_point_2, radius_point_left, end_angle_left)

    @property
    def shapes(self):
        return [self.cusp_right, self.cusp_left]

    def chop_axial(self, **kwargs):
        self.cusp_right.chop_axial(**kwargs)

    def chop_radial(self, **kwargs):
        self.cusp_right.chop_radial(**kwargs)
        self.cusp_left.chop_radial(**kwargs)

    def chop_tangential(self, **kwargs):
        self.cusp_right.chop_tangential(**kwargs)
        self.cusp_left.chop_tangential(**kwargs)

    def set_outer_patch(self, patch_name: str) -> None:
        self.cusp_left.set_outer_patch(patch_name)
        self.cusp_right.set_outer_patch(patch_name)

    def set_start_patch(self, patch_name: str) -> None:
        self.cusp_left.set_start_patch(patch_name)
        self.cusp_right.set_start_patch(patch_name)


class JointBase(Assembly, abc.ABC):
    def __init__(self, start_point: PointType, center_point: PointType, radius_point: PointType, branches: int = 4):
        start_point = np.asarray(start_point)
        center_point = np.asarray(center_point)
        radius_point = np.asarray(radius_point)

        self.assemblies: list[CuspCylinder] = []
        shapes: list[Shape] = []

        rotate_angles = self._get_angles(branches)
        rotation_axis = radius_point - start_point

        for i, angle in enumerate(rotate_angles):
            angle_right = (rotate_angles[(i + 1) % len(rotate_angles)] - angle) / 2
            angle_left = ((angle - rotate_angles[i - 1]) % (2 * np.pi)) / 2

            cylinder = CuspCylinder(start_point, center_point, radius_point, angle_left, angle_right)
            cylinder.rotate(angle, rotation_axis, center_point)
            self.assemblies.append(cylinder)
            shapes += cylinder.shapes

        super().__init__(shapes)

    @abc.abstractmethod
    def _get_angles(self, count: int) -> FloatListType:
        """Returns angles at which CuspCylinders must be rotated"""

    @property
    def center(self):
        # "center" is the start point
        return self.shapes[0].operations[0].top_face.points[0].position

    def chop_axial(self, **kwargs):
        for asm in self.assemblies:
            asm.chop_axial(**kwargs)

    def chop_radial(self, **kwargs):
        self.assemblies[0].chop_radial(**kwargs)

    def chop_tangential(self, **kwargs):
        self.assemblies[0].chop_tangential(**kwargs)
        self.assemblies[1].chop_tangential(**kwargs)

    def set_outer_patch(self, patch_name: str) -> None:
        for asm in self.assemblies:
            asm.set_outer_patch(patch_name)

    def set_hole_patch(self, hole: int, patch_name: str) -> None:
        self.assemblies[hole].set_start_patch(patch_name)


class NJoint(JointBase):
    def _get_angles(self, count):
        return np.linspace(0, 2 * np.pi, num=count, endpoint=False)


class TJoint(JointBase):
    def __init__(self, start_point: PointType, center_point: PointType, radius_point: PointType):
        super().__init__(start_point, center_point, radius_point)

    def _get_angles(self, _):
        return [0, np.pi / 2, 3 * np.pi / 2]


class LJoint(JointBase):
    def __init__(self, start_point: PointType, center_point: PointType, radius_point: PointType):
        super().__init__(start_point, center_point, radius_point)

    def _get_angles(self, _):
        return [0, np.pi / 2]

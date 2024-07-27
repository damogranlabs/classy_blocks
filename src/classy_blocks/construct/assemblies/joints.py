from typing import List

import numpy as np

from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.edges import Spline
from classy_blocks.construct.flat.sketches.disk import HalfDisk
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.shapes.cylinder import SemiCylinder
from classy_blocks.types import PointType
from classy_blocks.util import functions as f


class CuspSemiCylinder(SemiCylinder):
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
            edge_points = f.divide_arc(axis, axis_point_2, point_1, point_2, 5)
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


class TJoint(Assembly):
    def __init__(self, start_point: PointType, center_point: PointType, right_point: PointType, radius: float):
        start_point = np.array(start_point)
        center_point = np.array(center_point)
        right_point = np.array(right_point)

        start_vector = center_point - start_point
        right_vector = right_point - center_point

        left_point = center_point - right_vector

        radius_vector = radius * f.unit_vector(np.cross(right_vector, start_vector))

        # middle
        self.middle = CuspCylinder(start_point, center_point, start_point + radius_vector, np.pi / 4, np.pi / 4)

        # the right 'arm'
        self.right = CuspCylinder(right_point, center_point, right_point + radius_vector, np.pi / 4, np.pi / 2)

        # the left 'arm'
        self.left = CuspCylinder(left_point, center_point, left_point - radius_vector, np.pi / 4, np.pi / 2)

        super().__init__([*self.middle.shapes, *self.right.shapes, *self.left.shapes])


class NJoint(Assembly):
    def __init__(self, start_point: PointType, center_point: PointType, radius_point: PointType, branches: int = 4):
        start_point = np.asarray(start_point)
        center_point = np.asarray(center_point)
        radius_point = np.asarray(radius_point)

        self.assemblies: List[Assembly] = []
        shapes: List[Shape] = []

        cusp_angle = np.pi / branches
        base_asm = CuspCylinder(start_point, center_point, radius_point, cusp_angle, cusp_angle)
        rotate_angles = np.linspace(0, 2 * np.pi, num=branches, endpoint=False)
        rotation_axis = radius_point - start_point

        for angle in rotate_angles:
            asm = base_asm.copy().rotate(angle, rotation_axis, center_point)
            self.assemblies.append(asm)
            shapes += asm.shapes

        super().__init__(shapes)

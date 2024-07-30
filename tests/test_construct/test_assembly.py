import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.assemblies.joints import JointBase, LJoint, NJoint, TJoint
from classy_blocks.mesh import Mesh
from classy_blocks.types import NPPointListType
from classy_blocks.util import functions as f


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.center_point = [0, 0, 0]
        self.start_point = [0, -1, 0]
        self.radius_point = [0, -1, 0.3]

    def fanout(self, count: int) -> NPPointListType:
        angles = np.linspace(0, 2 * np.pi, num=count, endpoint=False)

        return np.array([f.rotate(self.start_point, angle, [0, 0, 1], self.center_point) for angle in angles])

    def get_joint_points(self, joint: JointBase) -> NPPointListType:
        points = []

        for i, shape in enumerate(joint.shapes):
            if i % 2 == 0:
                points.append(shape.operations[0].bottom_face.points[0].position)

        return np.array(points)

    def test_t_joint(self):
        joint = TJoint(self.start_point, self.center_point, self.radius_point)

        joint_points = self.get_joint_points(joint)
        expected_points = np.take(self.fanout(4), (0, 1, 3), axis=0)

        np.testing.assert_almost_equal(joint_points, expected_points)

    def test_l_joint(self):
        joint = LJoint(self.start_point, self.center_point, self.radius_point)

        joint_points = self.get_joint_points(joint)
        expected_points = np.take(self.fanout(4), (0, 1), axis=0)

        np.testing.assert_almost_equal(joint_points, expected_points)

    @parameterized.expand(((3,), (4,), (5,), (6,)))
    def test_n_joint(self, branches):
        joint = NJoint(self.start_point, self.center_point, self.radius_point, branches=branches)

        joint_points = self.get_joint_points(joint)
        expected_points = self.fanout(branches)

        np.testing.assert_almost_equal(joint_points, expected_points)

    def test_operations(self):
        joint = NJoint(self.start_point, self.center_point, self.radius_point, branches=3)

        self.assertEqual(len(joint.operations), 3 * 2 * 6)

    def test_center(self):
        joint = NJoint(self.start_point, self.center_point, self.radius_point, branches=3)

        np.testing.assert_equal(joint.center, self.center_point)

    def test_chop(self):
        mesh = Mesh()
        joint = TJoint(self.start_point, self.center_point, self.radius_point)

        cell_size = 0.1
        joint.chop_axial(start_size=cell_size)
        joint.chop_radial(start_size=cell_size)
        joint.chop_tangential(start_size=cell_size)

        mesh.add(joint)
        mesh.assemble()
        mesh.block_list.propagate_gradings()

    def test_set_patches(self):
        branches = 5

        joint = NJoint(self.start_point, self.center_point, self.radius_point, branches)

        joint.set_outer_patch("walls")

        for i in range(branches):
            joint.set_hole_patch(i, f"outlet_{i}")

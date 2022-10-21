import unittest

import numpy as np

from classy_blocks.classes.shapes import Elbow, ExtrudedRing


class TestElbow(unittest.TestCase):
    def setUp(self):
        center_point_1 = [0, 0, 0]
        radius_point_1 = [1, 0, 0]
        normal_1 = [0, 0, 1]

        sweep_angle = np.pi / 4
        arc_center = [0, 1, 0]
        rotation_axis = [0, 1, 0]
        radius_2 = 0.5

        self.elbow = Elbow(center_point_1, radius_point_1, normal_1, sweep_angle, arc_center, rotation_axis, radius_2)

    def test_elbow_operations(self):
        self.assertEqual(len(self.elbow.operations), 12)


class TestRing(unittest.TestCase):
    def setUp(self):
        self.axis_point_1 = [0, 0, 0]
        self.axis_point_2 = [0, 0, 1]
        self.inner_radius_point_1 = [0.5, 0, 0]
        self.outer_radius = 1

        self.n_segments = 4

        self.ring = ExtrudedRing(
            self.axis_point_1,
            self.axis_point_2,
            self.inner_radius_point_1,
            self.outer_radius,
            n_segments=self.n_segments,
        )

    def test_ring_operations(self):
        self.assertEqual(len(self.ring.operations), self.n_segments)

    def test_n_segments(self):
        n_segments = 8

        ring = ExtrudedRing(
            self.axis_point_1, self.axis_point_2, self.inner_radius_point_1, self.outer_radius, n_segments=n_segments
        )

        self.assertEqual(len(ring.operations), n_segments)

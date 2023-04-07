import unittest
import numpy as np

from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.util import functions as f


class ElbowTests(unittest.TestCase):
    """Tests of the Elbow shape"""

    def setUp(self):
        self.center_point_1 = f.vector(1.0, 1.0, 1.0)
        self.radius_point_1 = f.vector(2.0, 1.0, 1.0)
        self.normal_1 = f.vector(0, 1, 0)

        self.sweep_angle = -np.pi / 3

        self.arc_center = f.vector(3.0, 1.0, 1.0)
        self.rotation_axis = f.vector(1.0, 1.0, 2.0)

        self.radius_2 = 0.4

    @property
    def radius_1(self) -> float:
        """Start radius"""
        return f.norm(self.radius_point_1 - self.center_point_1)

    @property
    def elbow(self) -> Elbow:
        """The test subject"""
        return Elbow(
            self.center_point_1,
            self.radius_point_1,
            self.normal_1,
            self.sweep_angle,
            self.arc_center,
            self.rotation_axis,
            self.radius_2,
        )

    def test_radius_1(self):
        """Radius of the start sketch"""
        self.assertAlmostEqual(self.elbow.sketch_1.radius, self.radius_1)

    def test_radius_2(self):
        """Radius of the end sketch"""
        self.assertAlmostEqual(self.elbow.sketch_2.radius, self.radius_2)

    def test_radius_mid(self):
        """Radius of the middle sketch"""
        # should be between 1 and 2
        self.assertAlmostEqual(self.elbow.sketch_mid.radius, (self.radius_1 + self.radius_2) / 2)


class SphereTests(unittest.TestCase):
    # TODO: points, transforms
    pass
